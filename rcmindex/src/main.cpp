// rcmindex: overlapping-tile frame indexer for ROCm GPUs.
//
// Per batch of frames:
//   read thread    JPEG bytes from disk (parallel readers)
//   decode thread  rocJPEG hardware decode -> RGB planar in device memory
//   main thread    antialiased resize to 768x768 + normalize (HIP)
//                  -> ResNet-18 trunk once per frame (MIGraphX, fp16)
//                  -> 7x7 half-stride window pool + fc + L2 norm (HIP)
//                  -> async copy to pinned memory, write .npy
// Slots rotate between stages so disk, decode and compute overlap.

#include <rocjpeg/rocjpeg.h>

#include <migraphx/migraphx.hpp>

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
#include <iostream>
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

void hip(hipError_t status, std::source_location at = std::source_location::current()) {
    if (status != hipSuccess)
        throw std::runtime_error(std::format("{}:{}: HIP error: {}", at.file_name(), at.line(),
                                             hipGetErrorString(status)));
}

void jpeg(RocJpegStatus status, std::string_view what,
          std::source_location at = std::source_location::current()) {
    if (status != ROCJPEG_STATUS_SUCCESS)
        throw std::runtime_error(std::format("{}:{}: rocJPEG {} failed: {}", at.file_name(),
                                             at.line(), what, rocJpegGetErrorName(status)));
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// ---------------------------------------------------------------- options

struct Options {
    fs::path frames;
    fs::path model;
    fs::path output;  // <name>.npy; records and report sit beside it
    int batch = 32;
    int slots = 3;
    int readers = 8;
    std::size_t limit = 0;
    bool global = false;  // also write one whole-frame embedding per frame
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
        else if (a == "--global") o.global = true;
        else throw std::runtime_error(std::format("unknown argument {}", a));
    }
    if (o.frames.empty() || o.model.empty() || o.output.empty())
        throw std::runtime_error(
            "usage: rcmindex --frames DIR --model DIR --output OUT.npy [--batch 32] [--slots 3] "
            "[--readers 8] [--limit N] [--global]");
    if (o.slots < 2) throw std::runtime_error("--slots must be at least 2");
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
        hip(hipMalloc(&buffer, shapes[name].bytes()));
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

// ---------------------------------------------------------------- frames

struct Frames {
    std::vector<fs::path> paths;    // absolute
    std::vector<std::string> names; // relative to --frames, for records
};

Frames list_frames(const fs::path& root, std::size_t limit) {
    std::vector<fs::path> movies;
    for (const auto& entry : fs::directory_iterator(root))
        if (entry.is_directory()) movies.push_back(entry.path());
    std::ranges::sort(movies);
    Frames f;
    for (const auto& movie : movies) {
        std::vector<fs::path> jpgs;
        for (const auto& entry : fs::directory_iterator(movie))
            if (entry.path().extension() == ".jpg") jpgs.push_back(entry.path());
        std::ranges::sort(jpgs);
        for (auto& p : jpgs) {
            if (limit && f.paths.size() == limit) return f;
            f.names.push_back(fs::relative(p, root).generic_string());
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
    std::vector<std::vector<std::uint8_t>> bytes;  // encoded JPEGs
    std::vector<std::uint8_t*> decoded;            // device RGB planar per frame
    std::vector<std::size_t> capacity;
    std::vector<FrameView> views;
    FrameView* device_views = nullptr;
    __half* host_tiles = nullptr;   // pinned [batch, views, dim]
    __half* host_global = nullptr;  // pinned [batch, dim]
    hipEvent_t done{};
};

struct Timings {
    std::atomic<double> read{0}, decode{0}, compute{0}, wait{0}, write{0};
};

void add(std::atomic<double>& total, double value) {
    total.fetch_add(value, std::memory_order_relaxed);
}

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
                    for (int i = r; i < slot.count; i += o.readers) {
                        std::ifstream in(frames.paths[first + i], std::ios::binary);
                        if (!in) throw std::runtime_error("cannot read " + frames.paths[first + i].string());
                        slot.bytes[i].assign(std::istreambuf_iterator<char>(in), {});
                    }
                });
        }
        add(time.read, seconds_since(start));
        push(read_slots, s);
    }
    close(read_slots);
}

void decode_stage(const Frames& frames, std::vector<Slot>& slots, Channel<int>& read_slots,
                  Channel<int>& decoded_slots, Timings& time) {
    RocJpegHandle handle{};
    jpeg(rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, 0, &handle), "create");
    const int batch = int(slots.front().bytes.size());
    std::vector<RocJpegStreamHandle> streams(batch);
    for (auto& s : streams) jpeg(rocJpegStreamCreate(&s), "stream create");
    std::vector<RocJpegDecodeParams> params(batch);
    for (auto& p : params) p.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
    std::vector<RocJpegImage> images(batch);
    hipStream_t copy{};
    hip(hipStreamCreateWithFlags(&copy, hipStreamNonBlocking));

    while (const auto next = pop(read_slots)) {
        const auto start = Clock::now();
        Slot& slot = slots[*next];
        slot.max_height = 0;
        for (int i = 0; i < slot.count; ++i) {
            const auto& data = slot.bytes[i];
            const auto name = frames.names[slot.first + i];
            jpeg(rocJpegStreamParse(data.data(), data.size(), streams[i]), "parse " + name);
            std::uint8_t components = 0;
            RocJpegChromaSubsampling subsampling{};
            std::uint32_t widths[ROCJPEG_MAX_COMPONENT]{}, heights[ROCJPEG_MAX_COMPONENT]{};
            jpeg(rocJpegGetImageInfo(handle, streams[i], &components, &subsampling, widths, heights),
                 "info " + name);
            const std::uint32_t w = widths[0], h = heights[0];
            const std::size_t need = std::size_t(w) * h * 3;
            if (slot.capacity[i] < need) {
                if (slot.decoded[i]) hip(hipFree(slot.decoded[i]));
                hip(hipMalloc(&slot.decoded[i], need));
                slot.capacity[i] = need;
            }
            images[i] = {};
            for (int c = 0; c < 3; ++c) {
                images[i].channel[c] = slot.decoded[i] + std::size_t(c) * w * h;
                images[i].pitch[c] = w;
            }
            slot.views[i] = {.plane = {images[i].channel[0], images[i].channel[1], images[i].channel[2]},
                             .pitch = {w, w, w}, .width = w, .height = h};
            slot.max_height = std::max(slot.max_height, int(h));
        }
        // rocJPEG returns once the batch is decoded into `images`.
        jpeg(rocJpegDecodeBatched(handle, streams.data(), slot.count, params.data(), images.data()),
             "decode");
        hip(hipMemcpyAsync(slot.device_views, slot.views.data(), sizeof(FrameView) * slot.count,
                           hipMemcpyHostToDevice, copy));
        hip(hipStreamSynchronize(copy));
        add(time.decode, seconds_since(start));
        push(decoded_slots, *next);
    }
    close(decoded_slots);
    for (auto s : streams) rocJpegStreamDestroy(s);
    rocJpegDestroy(handle);
    hip(hipStreamDestroy(copy));
}

// ---------------------------------------------------------------- main

int run(const Options& o) {
    const auto started = Clock::now();
    const Head head = load_head(o.model);
    const int tile_input = head.grid * head.image_size;  // 768
    const int steps = 2 * head.grid - 1;                 // 7
    const int views = steps * steps;                     // 49

    const Frames frames = list_frames(o.frames, o.limit);
    if (frames.paths.empty()) throw std::runtime_error("no frames found");
    const std::size_t n = frames.paths.size();
    std::cout << std::format("{} frames, {} views each, batch {}\n", n, views, o.batch);

    fs::create_directories(o.output.parent_path());
    {
        std::ofstream records(fs::path(o.output).replace_extension(".records.json"));
        records << '[';
        for (std::size_t i = 0; i < n; ++i) records << (i ? ", " : "") << json_string(frames.names[i]);
        records << "]\n";
    }

    hipStream_t stream{};
    hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Trunk tiles = compile_trunk(o.model, o.batch, tile_input);
    std::optional<Trunk> whole;
    if (o.global) whole = compile_trunk(o.model, o.batch, head.image_size);
    const int window = tiles.side / head.grid;  // 6 cells per tile
    if (tiles.side % head.grid || window % 2)
        throw std::runtime_error(std::format("feature map {} does not tile into {}x{} half-stride windows",
                                             tiles.side, head.grid, head.grid));

    float* weight = nullptr;
    hip(hipMalloc(&weight, head.weights.size() * sizeof(float)));
    hip(hipMemcpy(weight, head.weights.data(), head.weights.size() * sizeof(float), hipMemcpyHostToDevice));
    const float* bias = weight + std::size_t(head.dim) * head.features;

    std::size_t max_frame_height = 2160;  // grows scratch on demand
    float* scratch = nullptr;
    hip(hipMalloc(&scratch, sizeof(float) * o.batch * 3 * max_frame_height * tile_input));
    __half* device_tiles = nullptr;
    __half* device_global = nullptr;
    hip(hipMalloc(&device_tiles, sizeof(__half) * o.batch * views * head.dim));
    hip(hipMalloc(&device_global, sizeof(__half) * o.batch * head.dim));

    std::vector<Slot> slots(o.slots);
    Channel<int> free_slots, read_slots, decoded_slots;
    for (int s = 0; s < o.slots; ++s) {
        Slot& slot = slots[s];
        slot.bytes.resize(o.batch);
        slot.decoded.assign(o.batch, nullptr);
        slot.capacity.assign(o.batch, 0);
        slot.views.resize(o.batch);
        hip(hipMalloc(&slot.device_views, sizeof(FrameView) * o.batch));
        hip(hipHostMalloc(&slot.host_tiles, sizeof(__half) * o.batch * views * head.dim));
        hip(hipHostMalloc(&slot.host_global, sizeof(__half) * o.batch * head.dim));
        hip(hipEventCreateWithFlags(&slot.done, hipEventDisableTiming));
        push(free_slots, s);
    }

    std::FILE* tile_file = std::fopen(o.output.c_str(), "wb");
    const std::array<std::size_t, 3> tile_shape{n, std::size_t(views), std::size_t(head.dim)};
    const std::size_t tile_data = write_npy_header(tile_file, tile_shape);
    std::FILE* global_file = nullptr;
    std::size_t global_data = 0;
    if (o.global) {
        const auto path = fs::path(o.output).replace_extension(".global.npy");
        global_file = std::fopen(path.c_str(), "wb");
        const std::array<std::size_t, 2> global_shape{n, std::size_t(head.dim)};
        global_data = write_npy_header(global_file, global_shape);
    }

    Timings time;
    std::jthread reader(read_stage, std::cref(frames), std::cref(o), std::ref(slots),
                        std::ref(free_slots), std::ref(read_slots), std::ref(time));
    std::jthread decoder(decode_stage, std::cref(frames), std::ref(slots), std::ref(read_slots),
                         std::ref(decoded_slots), std::ref(time));

    // Finish a slot: wait for its copies, write rows, hand it back to the reader.
    std::size_t written = 0;
    auto retire = [&](int s) {
        Slot& slot = slots[s];
        auto t = Clock::now();
        hip(hipEventSynchronize(slot.done));
        add(time.wait, seconds_since(t));
        t = Clock::now();
        const std::size_t row = std::size_t(views) * head.dim * sizeof(__half);
        std::fseek(tile_file, long(tile_data + slot.first * row), SEEK_SET);
        std::fwrite(slot.host_tiles, row, slot.count, tile_file);
        if (global_file) {
            const std::size_t grow = head.dim * sizeof(__half);
            std::fseek(global_file, long(global_data + slot.first * grow), SEEK_SET);
            std::fwrite(slot.host_global, grow, slot.count, global_file);
        }
        add(time.write, seconds_since(t));
        written += slot.count;
        if (written / 20'000 != (written - slot.count) / 20'000)
            std::cout << std::format("{}/{} frames, {:.1f}s\n", written, n, seconds_since(started))
                      << std::flush;
        push(free_slots, s);
    };

    std::optional<int> previous;
    while (const auto next = pop(decoded_slots)) {
        const auto t = Clock::now();
        Slot& slot = slots[*next];
        if (std::size_t(slot.max_height) > max_frame_height) {
            hip(hipStreamSynchronize(stream));
            hip(hipFree(scratch));
            max_frame_height = slot.max_height;
            hip(hipMalloc(&scratch, sizeof(float) * o.batch * 3 * max_frame_height * tile_input));
        }
        resize_normalize(slot.device_views, slot.count, slot.max_height, tile_input, scratch,
                         tiles.input, stream);
        const __half* features = run_trunk(tiles, stream);
        pool_project(features, slot.count, head.features, tiles.side, window, window / 2, steps,
                     weight, bias, head.dim, device_tiles, stream);
        hip(hipMemcpyAsync(slot.host_tiles, device_tiles, sizeof(__half) * slot.count * views * head.dim,
                           hipMemcpyDeviceToHost, stream));
        if (whole) {
            resize_normalize(slot.device_views, slot.count, slot.max_height, head.image_size, scratch,
                             whole->input, stream);
            const __half* global_features = run_trunk(*whole, stream);
            pool_project(global_features, slot.count, head.features, whole->side, whole->side, 1, 1,
                         weight, bias, head.dim, device_global, stream);
            hip(hipMemcpyAsync(slot.host_global, device_global, sizeof(__half) * slot.count * head.dim,
                               hipMemcpyDeviceToHost, stream));
        }
        hip(hipEventRecord(slot.done, stream));
        add(time.compute, seconds_since(t));
        // The GPU now works on this slot while we write out the previous one.
        if (previous) retire(*previous);
        previous = *next;
    }
    if (previous) retire(*previous);
    close(free_slots);
    reader.join();
    decoder.join();
    std::fclose(tile_file);
    if (global_file) std::fclose(global_file);

    const double elapsed = seconds_since(started);
    hipDeviceProp_t props{};
    hip(hipGetDeviceProperties(&props, 0));
    std::ofstream report(fs::path(o.output).replace_extension(".json"));
    report << std::format(
        "{{\n  \"gpu\": {},\n  \"frames\": {},\n  \"views_per_frame\": {},\n  \"embeddings\": {},\n"
        "  \"global\": {},\n  \"elapsed_seconds\": {:.3f},\n  \"frames_per_second\": {:.1f},\n"
        "  \"views_per_second\": {:.1f},\n  \"batch_size\": {},\n  \"slots\": {},\n  \"readers\": {},\n"
        "  \"encoder\": \"migraphx_fp16_shared_trunk\",\n  \"decoder\": \"rocjpeg\",\n"
        "  \"stage_seconds\": {{\"read\": {:.3f}, \"decode\": {:.3f}, \"enqueue\": {:.3f}, "
        "\"gpu_wait\": {:.3f}, \"write\": {:.3f}}}\n}}\n",
        json_string(props.name), n, views, n * views, o.global, elapsed, n / elapsed,
        n * views / elapsed, o.batch, o.slots, o.readers, time.read.load(), time.decode.load(),
        time.compute.load(), time.wait.load(), time.write.load());
    std::cout << std::format("indexed {} frames in {:.1f}s ({:.0f} frames/s, {:.0f} views/s)\n", n,
                             elapsed, n / elapsed, n * views / elapsed);
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
