// rcmindex: overlapping-tile frame indexer for ROCm and CUDA GPUs.
//
// Per batch of frames:
//   read thread    JPEG/WebP read + CPU decode into pinned memory (parallel)
//   upload thread  pinned RGB -> device memory
//   main thread    antialiased resize to 768x768 + normalize (HIP/CUDA kernel)
//                  -> ResNet-18 trunk once per frame (MIGraphX or TensorRT, fp16)
//                  -> 7x7 half-stride window pool + fc + L2 norm (HIP/CUDA kernel)
//                  -> async copy to pinned memory, write .npy
// Slots rotate between stages so disk, decode and compute overlap.

#include <turbojpeg.h>
#include <webp/decode.h>

#if defined(RCM_CUDA)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#else
#include <migraphx/migraphx.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "kernels.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace rcm {
namespace {

// ---------------------------------------------------------------- errors

void gpu(hipError_t status, std::source_location at = std::source_location::current()) {
    if (status != hipSuccess)
        throw std::runtime_error(std::format("{}:{}: GPU error: {}", at.file_name(), at.line(),
                                             hipGetErrorString(status)));
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// ---------------------------------------------------------------- options

struct Options {
    fs::path frames;
    fs::path model;
    fs::path output;  // <name>.npy, or a directory with --movies-from
    bool movies_from_stdin = false;  // index movie dirs as they arrive on stdin
    int batch = 32;
    int slots = 3;
    int readers = 16;  // CPU decode threads
    std::size_t limit = 0;
    // Tile grids to index. Grid g runs the trunk at g*192 so each tile is 192 px
    // (the query scale) and pools (2g-1)^2 half-stride tiles; grid 1 is the
    // whole frame. Files: 4 -> <movie>.npy, 1 -> .global.npy, g -> .g<g>.npy.
    std::vector<int> grids{4};
};

Options parse(std::span<char*> args) {
    Options o;
    auto value = [&](std::size_t& i) -> std::string {
        if (i + 1 >= args.size()) throw std::runtime_error(std::format("{} needs a value", args[i]));
        return args[++i];
    };
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--frames") o.frames = value(i);
        else if (a == "--model") o.model = value(i);
        else if (a == "--output") o.output = value(i);
        else if (a == "--batch") o.batch = std::stoi(value(i));
        else if (a == "--slots") o.slots = std::stoi(value(i));
        else if (a == "--readers") o.readers = std::stoi(value(i));
        else if (a == "--limit") o.limit = std::stoull(value(i));
        else if (a == "--global") o.grids.push_back(1);
        else if (a == "--grids") {
            o.grids.clear();
            std::string list = value(i);
            for (const auto part : std::views::split(list, ','))
                o.grids.push_back(std::stoi(std::string(part.begin(), part.end())));
        }
        else if (a == "--movies-from") {
            if (value(i) != "-") throw std::runtime_error("--movies-from only supports '-' (stdin)");
            o.movies_from_stdin = true;
        }
        else throw std::runtime_error(std::format("unknown argument {}", a));
    }
    if ((o.frames.empty() && !o.movies_from_stdin) || o.model.empty() || o.output.empty())
        throw std::runtime_error(
            "usage: rcmindex --model DIR (--frames DIR --output OUT.npy | --movies-from - --output DIR)\n"
            "                [--grids 4,3,2,1] [--global] [--batch 32] [--slots 3] [--readers 16] [--limit N]");
    if (o.slots < 2) throw std::runtime_error("--slots must be at least 2");
    std::ranges::sort(o.grids, std::greater{});
    o.grids.erase(std::ranges::unique(o.grids).begin(), o.grids.end());
    if (o.grids.empty() || o.grids.back() < 1) throw std::runtime_error("--grids needs positive sizes");
    return o;
}

// ---------------------------------------------------------------- model files

struct Head {
    int dim = 0;
    int features = 0;
    int image_size = 0;
    int grid = 0;
    std::vector<float> weights;  // [dim * features] then [dim] bias
};

int json_int(std::string_view text, std::string_view key) {
    const auto at = text.find(std::format("\"{}\"", key));
    if (at == std::string_view::npos) throw std::runtime_error(std::format("head.json lacks {}", key));
    const auto colon = text.find(':', at);
    return std::stoi(std::string(text.substr(colon + 1, 32)));
}

std::string slurp(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::format("cannot read {}", path.string()));
    return {std::istreambuf_iterator<char>(in), {}};
}

Head load_head(const fs::path& dir) {
    const std::string meta = slurp(dir / "head.json");
    Head h{.dim = json_int(meta, "embedding_dim"),
           .features = json_int(meta, "features"),
           .image_size = json_int(meta, "image_size"),
           .grid = json_int(meta, "grid"),
           .weights = {}};
    const std::string raw = slurp(dir / "head.bin");
    const std::size_t expected = sizeof(float) * std::size_t(h.dim) * (h.features + 1);
    if (raw.size() != expected)
        throw std::runtime_error(std::format("head.bin is {} bytes, expected {}", raw.size(), expected));
    h.weights.resize(raw.size() / sizeof(float));
    std::memcpy(h.weights.data(), raw.data(), raw.size());
    return h;
}

#if defined(RCM_CUDA)

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};
TrtLogger trt_logger;

// trunk.onnx is already fp16 (export_model.py), so a strongly typed network runs
// it in fp16 as is; TensorRT 10 deprecates the kFP16 builder flag for this.
#if NV_TENSORRT_MAJOR >= 10
constexpr std::uint32_t kNetworkFlags = 1U << int(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
constexpr std::uint32_t kNetworkFlags = 1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif

// A TensorRT engine built for one batch and input size, with its device buffers.
struct Trunk {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;  // declared last: destroyed first
    __half* input = nullptr;
    const __half* output = nullptr;
    std::vector<void*> owned;
    int side = 0;  // output feature map side
};

Trunk compile_trunk(const fs::path& model_dir, int batch, int size) {
    // Building takes a minute or two; cache the engine. A .plan only loads on
    // the GPU model and TensorRT version that built it: delete it after either changes.
    const fs::path cache = model_dir / std::format("trunk_b{}_s{}.plan", batch, size);
    const nvinfer1::Dims4 shape{batch, 3, size, size};
    if (!fs::exists(cache)) {
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(trt_logger));
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(kNetworkFlags));
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, trt_logger));
        if (!parser->parseFromFile((model_dir / "trunk.onnx").c_str(), int(nvinfer1::ILogger::Severity::kWARNING)))
            throw std::runtime_error("TensorRT cannot parse trunk.onnx");
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
#if NV_TENSORRT_MAJOR < 10
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
        // trunk.onnx has dynamic batch/height/width; build for exactly this shape.
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (const auto which : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT,
                                 nvinfer1::OptProfileSelector::kMAX})
            profile->setDimensions("images", which, shape);
        config->addOptimizationProfile(profile);
        std::cerr << std::format("building TensorRT engine for batch {} at {}x{} (one-off)\n", batch, size, size);
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) throw std::runtime_error("TensorRT engine build failed");
        std::ofstream(cache, std::ios::binary).write(static_cast<const char*>(plan->data()), plan->size());
    }
    Trunk t;
    const std::string plan = slurp(cache);
    t.runtime.reset(nvinfer1::createInferRuntime(trt_logger));
    t.engine.reset(t.runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!t.engine) throw std::runtime_error(std::format("cannot load {}; delete it to rebuild", cache.string()));
    t.context.reset(t.engine->createExecutionContext());
    if (!t.context->setInputShape("images", shape)) throw std::runtime_error("trunk.onnx has no 'images' input");
    for (int i = 0; i < t.engine->getNbIOTensors(); ++i) {
        const char* name = t.engine->getIOTensorName(i);
        if (t.engine->getTensorDataType(name) != nvinfer1::DataType::kHALF)
            throw std::runtime_error(std::format("trunk tensor {} is not fp16", name));
        const nvinfer1::Dims dims = t.context->getTensorShape(name);
        std::size_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d) count *= std::size_t(dims.d[d]);
        void* buffer = nullptr;
        gpu(hipMalloc(&buffer, count * sizeof(__half)));
        t.owned.push_back(buffer);
        t.context->setTensorAddress(name, buffer);
        if (t.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            t.input = static_cast<__half*>(buffer);
        } else {
            t.output = static_cast<const __half*>(buffer);
            t.side = int(dims.d[3]);
        }
    }
    if (!t.input || !t.output) throw std::runtime_error("trunk.onnx needs one input and one output");
    return t;
}

const __half* run_trunk(Trunk& t, hipStream_t stream) {
    if (!t.context->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueue failed");
    return t.output;
}

constexpr const char* kEncoder = "tensorrt_fp16_shared_trunk";

#else

// A trunk compiled for one batch and input size, with its device buffers.
struct Trunk {
    migraphx::program program;
    migraphx::program_parameters parameters;
    __half* input = nullptr;
    std::vector<void*> owned;
    int side = 0;  // output feature map side
};

Trunk compile_trunk(const fs::path& model_dir, int batch, int size) {
    // Compiling takes minutes on MI300-class GPUs; cache the compiled program.
    const fs::path cache = model_dir / std::format("trunk_b{}_s{}.mxr", batch, size);
    Trunk t;
    if (fs::exists(cache)) {
        t.program = migraphx::load(cache.c_str());
    } else {
        migraphx::onnx_options onnx;
        onnx.set_input_parameter_shape("images", {std::size_t(batch), 3, std::size_t(size), std::size_t(size)});
        t.program = migraphx::parse_onnx((model_dir / "trunk.onnx").c_str(), onnx);
        migraphx::compile_options compile;
        compile.set_offload_copy(false);  // we own device buffers; no host round trips
        t.program.compile(migraphx::target("gpu"), compile);
        migraphx::save(t.program, cache.c_str());
    }
    auto shapes = t.program.get_parameter_shapes();
    for (const char* name : shapes.names()) {
        void* buffer = nullptr;
        gpu(hipMalloc(&buffer, shapes[name].bytes()));
        t.owned.push_back(buffer);
        if (std::string_view(name) == "images") t.input = static_cast<__half*>(buffer);
        t.parameters.add(name, migraphx::argument(shapes[name], buffer));
    }
    if (!t.input) throw std::runtime_error("trunk.onnx has no 'images' input");
    const auto out = t.program.get_output_shapes()[0].lengths();
    t.side = int(out.at(3));
    return t;
}

const __half* run_trunk(Trunk& t, hipStream_t stream) {
    // MIGraphX schedules its own fused kernels; capturing it in a hipGraph is
    // not supported, but a compiled program already launches few kernels.
    auto results = t.program.run_async(t.parameters, stream);
    return reinterpret_cast<const __half*>(results[0].data());
}

constexpr const char* kEncoder = "migraphx_fp16_shared_trunk";

#endif

// ---------------------------------------------------------------- frames

struct Frames {
    std::vector<fs::path> paths;    // absolute
    std::vector<std::string> names; // relative to --frames, for records
};

// Frames directly in `root` and in its movie subdirectories, sorted: JPEGs,
// or WebP copies (.local/frames-small/) where the original was pruned; a JPEG
// wins over a WebP of the same name. Record names are relative to `base` and
// always end in .jpg, the frame's catalog name, so a copy indexes as its original.
Frames list_frames(const fs::path& root, const fs::path& base, std::size_t limit) {
    std::vector<fs::path> dirs{root};
    for (const auto& entry : fs::directory_iterator(root))
        if (entry.is_directory()) dirs.push_back(entry.path());
    std::ranges::sort(dirs);
    Frames f;
    for (const auto& dir : dirs) {
        std::map<std::string, fs::path> frames;  // stem -> file, sorted by stem
        for (const auto& entry : fs::directory_iterator(dir)) {
            const fs::path& p = entry.path();
            if (!entry.is_regular_file() || (p.extension() != ".jpg" && p.extension() != ".webp")) continue;
            auto [at, added] = frames.try_emplace(p.stem().string(), p);
            if (!added && p.extension() == ".jpg") at->second = p;
        }
        for (auto& [stem, p] : frames) {
            if (limit && f.paths.size() == limit) return f;
            f.names.push_back(fs::relative(p, base).replace_extension(".jpg").generic_string());
            f.paths.push_back(std::move(p));
        }
    }
    return f;
}

std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + '"';
}

// ---------------------------------------------------------------- npy output

// Writes an fp16 .npy of `shape`; returns the byte offset of the data.
std::size_t write_npy_header(std::FILE* file, std::span<const std::size_t> shape) {
    std::string dims;
    for (auto d : shape) dims += std::format("{}, ", d);
    if (shape.size() > 1) dims.pop_back(), dims.pop_back();
    std::string header =
        std::format("{{'descr': '<f2', 'fortran_order': False, 'shape': ({}), }}", dims);
    const std::size_t total = (10 + header.size() + 1 + 63) / 64 * 64;
    header.append(total - 10 - header.size() - 1, ' ');
    header += '\n';
    const std::uint16_t length = std::uint16_t(header.size());
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, file);
    std::fwrite(&length, 2, 1, file);
    std::fwrite(header.data(), 1, header.size(), file);
    return total;
}

// ---------------------------------------------------------------- channels

template <typename T>
struct Channel {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<T> items;
    bool closed = false;
};

template <typename T>
void push(Channel<T>& c, T item) {
    {
        std::lock_guard lock(c.mutex);
        c.items.push_back(std::move(item));
    }
    c.ready.notify_one();
}

template <typename T>
void close(Channel<T>& c) {
    {
        std::lock_guard lock(c.mutex);
        c.closed = true;
    }
    c.ready.notify_all();
}

template <typename T>
std::optional<T> pop(Channel<T>& c) {
    std::unique_lock lock(c.mutex);
    c.ready.wait(lock, [&] { return !c.items.empty() || c.closed; });
    if (c.items.empty()) return std::nullopt;
    T item = std::move(c.items.front());
    c.items.pop_front();
    return item;
}

// ---------------------------------------------------------------- slots

struct Slot {
    std::size_t first = 0;  // index of the batch's first frame
    int count = 0;
    int max_height = 0;
    std::vector<std::vector<std::uint8_t>> bytes;  // encoded JPEG/WebP files
    std::vector<std::uint8_t*> host_rgb;           // pinned, decoded RGB per frame
    std::vector<std::size_t> host_capacity;
    std::vector<std::uint8_t*> decoded;            // device RGB (interleaved) per frame
    std::vector<std::size_t> capacity;
    std::vector<FrameView> views;
    FrameView* device_views = nullptr;
    std::vector<__half*> host_out;  // pinned [batch, views, dim] per scale
    hipEvent_t done{};
};

struct Timings {
    std::atomic<double> read{0}, decode{0}, compute{0}, wait{0}, write{0};
};

void add(std::atomic<double>& total, double value) {
    total.fetch_add(value, std::memory_order_relaxed);
}

// Reader threads read and decode JPEGs (libjpeg-turbo) and WebPs (libwebp) on
// the CPU straight into pinned memory. GPU JPEG decoding (rocJPEG) is avoided: its VA-API
// driver loads a second LLVM next to MIGraphX's and aborts the process.
void read_stage(const Frames& frames, const Options& o, std::vector<Slot>& slots,
                Channel<int>& free_slots, Channel<int>& read_slots, Timings& time) {
    for (std::size_t first = 0; first < frames.paths.size(); first += o.batch) {
        const int s = *pop(free_slots);
        const auto start = Clock::now();
        Slot& slot = slots[s];
        slot.first = first;
        slot.count = int(std::min<std::size_t>(o.batch, frames.paths.size() - first));
        {
            std::vector<std::jthread> readers;
            for (int r = 0; r < o.readers; ++r)
                readers.emplace_back([&, r] {
                    tjhandle decoder = tjInitDecompress();
                    for (int i = r; i < slot.count; i += o.readers) {
                        const auto& path = frames.paths[first + i];
                        std::ifstream in(path, std::ios::binary);
                        if (!in) throw std::runtime_error("cannot read " + path.string());
                        auto& bytes = slot.bytes[i];
                        bytes.assign(std::istreambuf_iterator<char>(in), {});
                        const bool webp = path.extension() == ".webp";
                        int w = 0, h = 0, subsampling = 0, colorspace = 0;
                        if (webp ? !WebPGetInfo(bytes.data(), bytes.size(), &w, &h)
                                 : tjDecompressHeader3(decoder, bytes.data(), bytes.size(), &w, &h, &subsampling,
                                                       &colorspace) != 0)
                            throw std::runtime_error(std::format("{}: unreadable header", path.string()));
                        const std::size_t need = std::size_t(w) * h * 3;
                        if (slot.host_capacity[i] < need) {
                            if (slot.host_rgb[i]) gpu(hipHostFree(slot.host_rgb[i]));
                            gpu(hipHostMalloc(&slot.host_rgb[i], need));
                            slot.host_capacity[i] = need;
                        }
                        if (webp ? !WebPDecodeRGBInto(bytes.data(), bytes.size(), slot.host_rgb[i], need, w * 3)
                                 : tjDecompress2(decoder, bytes.data(), bytes.size(), slot.host_rgb[i], w, w * 3, h,
                                                 TJPF_RGB, 0) != 0)
                            throw std::runtime_error(std::format("{}: cannot decode", path.string()));
                        slot.views[i].width = std::uint32_t(w);
                        slot.views[i].height = std::uint32_t(h);
                    }
                    tjDestroy(decoder);
                });
        }
        add(time.read, seconds_since(start));
        push(read_slots, s);
    }
    close(read_slots);
}

// Copy decoded frames to the GPU while the previous batch computes.
void upload_stage(std::vector<Slot>& slots, Channel<int>& read_slots, Channel<int>& decoded_slots,
                  Timings& time) {
    hipStream_t copy{};
    gpu(hipStreamCreateWithFlags(&copy, hipStreamNonBlocking));
    while (const auto next = pop(read_slots)) {
        const auto start = Clock::now();
        Slot& slot = slots[*next];
        slot.max_height = 0;
        for (int i = 0; i < slot.count; ++i) {
            FrameView& view = slot.views[i];
            const std::size_t need = std::size_t(view.width) * view.height * 3;
            if (slot.capacity[i] < need) {
                if (slot.decoded[i]) gpu(hipFree(slot.decoded[i]));
                gpu(hipMalloc(&slot.decoded[i], need));
                slot.capacity[i] = need;
            }
            gpu(hipMemcpyAsync(slot.decoded[i], slot.host_rgb[i], need, hipMemcpyHostToDevice, copy));
            for (int c = 0; c < 3; ++c) {
                view.plane[c] = slot.decoded[i] + c;  // interleaved RGB
                view.pitch[c] = view.width * 3;
            }
            view.step = 3;
            slot.max_height = std::max(slot.max_height, int(view.height));
        }
        gpu(hipMemcpyAsync(slot.device_views, slot.views.data(), sizeof(FrameView) * slot.count,
                           hipMemcpyHostToDevice, copy));
        gpu(hipStreamSynchronize(copy));
        add(time.decode, seconds_since(start));
        push(decoded_slots, *next);
    }
    close(decoded_slots);
    gpu(hipStreamDestroy(copy));
}

// ---------------------------------------------------------------- engine

// One tile grid: its own trunk resolution, windows and output file.
struct Scale {
    int grid = 0;
    int input = 0;   // trunk input side: grid * 192
    int steps = 0;   // windows per side: 2*grid-1 (1 for the whole frame)
    int window = 0;  // feature cells per tile
    int views = 0;   // steps * steps
    Trunk trunk;
    __half* device_out = nullptr;
    std::string suffix;  // output file suffix
};

// Everything that lives across movies: compiled trunks, weights, buffers, slots.
struct Engine {
    Options options;
    Head head;
    std::vector<Scale> scales;
    int max_input = 0;
    hipStream_t stream{};
    float* weight = nullptr;
    const float* bias = nullptr;
    std::size_t max_frame_height = 2160;
    float* scratch = nullptr;
    std::vector<Slot> slots;
};

void setup(Engine& e, const Options& o) {
    e.options = o;
    e.head = load_head(o.model);
    gpu(hipStreamCreateWithFlags(&e.stream, hipStreamNonBlocking));
    for (const int g : o.grids) {
        Scale s;
        s.grid = g;
        s.input = g * e.head.image_size;
        s.steps = g == 1 ? 1 : 2 * g - 1;
        s.trunk = compile_trunk(o.model, o.batch, s.input);
        s.window = s.trunk.side / g;
        if (s.trunk.side % g || (g > 1 && s.window % 2))
            throw std::runtime_error(std::format("feature map {} does not tile into {}x{} half-stride windows",
                                                 s.trunk.side, g, g));
        s.views = s.steps * s.steps;
        s.suffix = g == e.head.grid ? ".npy" : g == 1 ? ".global.npy" : std::format(".g{}.npy", g);
        gpu(hipMalloc(&s.device_out, sizeof(__half) * o.batch * s.views * e.head.dim));
        e.max_input = std::max(e.max_input, s.input);
        e.scales.push_back(std::move(s));
    }
    const auto& w = e.head.weights;
    gpu(hipMalloc(&e.weight, w.size() * sizeof(float)));
    gpu(hipMemcpy(e.weight, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice));
    e.bias = e.weight + std::size_t(e.head.dim) * e.head.features;
    gpu(hipMalloc(&e.scratch, sizeof(float) * o.batch * 3 * e.max_frame_height * e.max_input));
    e.slots.resize(o.slots);
    for (Slot& slot : e.slots) {
        slot.bytes.resize(o.batch);
        slot.host_rgb.assign(o.batch, nullptr);
        slot.host_capacity.assign(o.batch, 0);
        slot.decoded.assign(o.batch, nullptr);
        slot.capacity.assign(o.batch, 0);
        slot.views.resize(o.batch);
        gpu(hipMalloc(&slot.device_views, sizeof(FrameView) * o.batch));
        for (const Scale& s : e.scales) {
            __half* host = nullptr;
            gpu(hipHostMalloc(&host, sizeof(__half) * o.batch * s.views * e.head.dim));
            slot.host_out.push_back(host);
        }
        gpu(hipEventCreateWithFlags(&slot.done, hipEventDisableTiming));
    }
}

fs::path with_suffix(const fs::path& base, std::string_view suffix) {
    return fs::path(base).replace_extension().string() + std::string(suffix);
}

// Index one frame list. `base` is <dir>/<movie>.npy; each grid writes its own
// .npy next to it, plus .records.json and a .json report. Returns frames/s.
double index_frames(Engine& e, const Frames& frames, const fs::path& base) {
    const Options& o = e.options;
    const auto started = Clock::now();
    const std::size_t n = frames.paths.size();
    const int dim = e.head.dim;

    if (base.has_parent_path()) fs::create_directories(base.parent_path());
    {
        std::ofstream records(with_suffix(base, ".records.json"));
        records << '[';
        for (std::size_t i = 0; i < n; ++i) records << (i ? ", " : "") << json_string(frames.names[i]);
        records << "]\n";
    }

    struct Output {
        std::FILE* file;
        std::size_t data;  // byte offset of the array
        std::size_t row;   // bytes per frame
    };
    std::vector<Output> outputs;
    for (const Scale& s : e.scales) {
        const fs::path path = with_suffix(base, s.suffix);
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) throw std::runtime_error("cannot write " + path.string());
        std::size_t data = 0;
        if (s.grid == 1) {
            const std::array<std::size_t, 2> shape{n, std::size_t(dim)};
            data = write_npy_header(file, shape);
        } else {
            const std::array<std::size_t, 3> shape{n, std::size_t(s.views), std::size_t(dim)};
            data = write_npy_header(file, shape);
        }
        outputs.push_back({file, data, std::size_t(s.views) * dim * sizeof(__half)});
    }

    Channel<int> free_slots, read_slots, decoded_slots;
    for (int s = 0; s < o.slots; ++s) push(free_slots, s);
    Timings time;
    std::jthread reader(read_stage, std::cref(frames), std::cref(o), std::ref(e.slots),
                        std::ref(free_slots), std::ref(read_slots), std::ref(time));
    std::jthread decoder(upload_stage, std::ref(e.slots), std::ref(read_slots), std::ref(decoded_slots),
                         std::ref(time));

    // Finish a slot: wait for its copies, write rows, hand it back to the reader.
    std::size_t written = 0;
    auto retire = [&](int s) {
        Slot& slot = e.slots[s];
        auto t = Clock::now();
        gpu(hipEventSynchronize(slot.done));
        add(time.wait, seconds_since(t));
        t = Clock::now();
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            std::fseek(outputs[k].file, long(outputs[k].data + slot.first * outputs[k].row), SEEK_SET);
            std::fwrite(slot.host_out[k], outputs[k].row, slot.count, outputs[k].file);
        }
        add(time.write, seconds_since(t));
        written += slot.count;
        if (written / 20'000 != (written - slot.count) / 20'000)
            std::cerr << std::format("  {}/{} frames, {:.1f}s\n", written, n, seconds_since(started));
        push(free_slots, s);
    };

    std::optional<int> previous;
    while (const auto next = pop(decoded_slots)) {
        const auto t = Clock::now();
        Slot& slot = e.slots[*next];
        if (std::size_t(slot.max_height) > e.max_frame_height) {
            gpu(hipStreamSynchronize(e.stream));
            gpu(hipFree(e.scratch));
            e.max_frame_height = slot.max_height;
            gpu(hipMalloc(&e.scratch, sizeof(float) * o.batch * 3 * e.max_frame_height * e.max_input));
        }
        for (std::size_t k = 0; k < e.scales.size(); ++k) {
            Scale& s = e.scales[k];
            resize_normalize(slot.device_views, slot.count, slot.max_height, s.input, e.scratch,
                             s.trunk.input, e.stream);
            const __half* features = run_trunk(s.trunk, e.stream);
            pool_project(features, slot.count, e.head.features, s.trunk.side, s.window,
                         s.grid == 1 ? s.window : s.window / 2, s.steps, e.weight, e.bias, dim,
                         s.device_out, e.stream);
            gpu(hipMemcpyAsync(slot.host_out[k], s.device_out, sizeof(__half) * slot.count * s.views * dim,
                               hipMemcpyDeviceToHost, e.stream));
        }
        gpu(hipEventRecord(slot.done, e.stream));
        add(time.compute, seconds_since(t));
        // The GPU now works on this slot while we write out the previous one.
        if (previous) retire(*previous);
        previous = *next;
    }
    if (previous) retire(*previous);
    close(free_slots);
    reader.join();
    decoder.join();
    for (const Output& out : outputs) std::fclose(out.file);

    const double elapsed = seconds_since(started);
    int views = 0;
    std::string grids;
    for (const Scale& s : e.scales) {
        views += s.views;
        grids += std::format("{}{}", grids.empty() ? "" : ", ", s.grid);
    }
    hipDeviceProp_t props{};
    gpu(hipGetDeviceProperties(&props, 0));
    std::ofstream report(with_suffix(base, ".json"));
    report << std::format(
        "{{\n  \"gpu\": {},\n  \"frames\": {},\n  \"grids\": [{}],\n  \"views_per_frame\": {},\n"
        "  \"embeddings\": {},\n  \"elapsed_seconds\": {:.3f},\n  \"frames_per_second\": {:.1f},\n"
        "  \"views_per_second\": {:.1f},\n  \"batch_size\": {},\n  \"slots\": {},\n  \"readers\": {},\n"
        "  \"encoder\": {},\n  \"decoder\": \"libjpeg-turbo, libwebp\",\n"
        "  \"stage_seconds\": {{\"read_decode\": {:.3f}, \"upload\": {:.3f}, \"enqueue\": {:.3f}, "
        "\"gpu_wait\": {:.3f}, \"write\": {:.3f}}}\n}}\n",
        json_string(props.name), n, grids, views, n * views, elapsed, n / elapsed, n * views / elapsed,
        o.batch, o.slots, o.readers, json_string(kEncoder), time.read.load(), time.decode.load(), time.compute.load(),
        time.wait.load(), time.write.load());
    return n / elapsed;
}

int run(const Options& o) {
    Engine e;
    setup(e, o);  // compiles or loads the trunk once, then reused for every movie

    if (!o.movies_from_stdin) {
        const Frames frames = list_frames(o.frames, o.frames, o.limit);
        if (frames.paths.empty()) throw std::runtime_error("no frames found");
        std::cout << std::format("{} frames, {} grids, batch {}\n", frames.paths.size(), e.scales.size(), o.batch);
        const double fps = index_frames(e, frames, o.output);
        std::cout << std::format("indexed {} frames ({:.0f} frames/s)\n", frames.paths.size(), fps);
        return 0;
    }

    // Streaming mode: a fetcher prints each movie directory once it is fully
    // extracted; index it immediately into <output>/<collection>/<movie>.npy.
    const auto started = Clock::now();
    std::size_t movies = 0, total = 0;
    for (std::string line; std::getline(std::cin, line);) {
        if (line.empty()) continue;
        const fs::path dir = fs::path(line).lexically_normal();
        const std::string movie = dir.filename().string();
        const Frames frames = list_frames(dir, dir.parent_path(), o.limit);
        if (frames.paths.empty()) {
            std::cerr << std::format("skip {}: no frames\n", movie);
            continue;
        }
        // <output>/<collection>/<movie>.npy, collection being the fetch parent dir.
        const fs::path out = o.output / dir.parent_path().filename() / (movie + ".npy");
        const double fps = index_frames(e, frames, out);
        ++movies;
        total += frames.paths.size();
        // Machine-readable line for the driver script (e.g. to delete the frames).
        std::cout << std::format("DONE {} {} {:.0f}\n", dir.string(), frames.paths.size(), fps) << std::flush;
        std::cerr << std::format("{} {}: {} frames at {:.0f} frames/s; {} frames in {:.0f}s overall\n",
                                 movies, movie, frames.paths.size(), fps, total, seconds_since(started));
    }
    return 0;
}

}  // namespace
}  // namespace rcm

int main(int argc, char** argv) {
    try {
        return rcm::run(rcm::parse(std::span(argv, argc)));
    } catch (const std::exception& error) {
        std::cerr << "rcmindex: " << error.what() << '\n';
        return 1;
    }
}
