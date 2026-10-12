#include "kernels.hpp"

namespace rcm {
namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

// Source span and weight of output pixel `o` along one axis (PyTorch's
// antialiased bilinear: a triangle filter widened by the downscale factor).
struct Taps {
    int first;
    int count;
    float center;
    float inv_scale;
};

__device__ Taps taps_for(int o, int in_size, int out_size) {
    const float scale = float(in_size) / float(out_size);
    const float support = scale >= 1.0f ? scale : 1.0f;
    const float center = scale * (o + 0.5f);
    const int first = max(int(center - support + 0.5f), 0);
    const int last = min(int(center + support + 0.5f), in_size);
    return {first, last - first, center, scale >= 1.0f ? 1.0f / scale : 1.0f};
}

__device__ float tap_weight(const Taps& t, int j) {
    const float x = (j + t.first - t.center + 0.5f) * t.inv_scale;
    return fmaxf(0.0f, 1.0f - fabsf(x));
}

// Pass 1: uint8 planes -> horizontally resized floats [frame, c, y, size].
__global__ void horizontal(const FrameView* frames, int max_height, int size, float* scratch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int fc = blockIdx.z;  // frame * 3 + channel
    const FrameView f = frames[fc / 3];
    const int c = fc % 3;
    if (x >= size || y >= int(f.height)) return;
    const Taps t = taps_for(x, f.width, size);
    const std::uint8_t* row = f.plane[c] + std::size_t(y) * f.pitch[c] + std::size_t(t.first) * f.step;
    float sum = 0.0f, total = 0.0f;
    for (int j = 0; j < t.count; ++j) {
        const float w = tap_weight(t, j);
        sum += w * row[std::size_t(j) * f.step];
        total += w;
    }
    scratch[(std::size_t(fc) * max_height + y) * size + x] = sum / total;
}

// Pass 2: vertical resize + ImageNet normalization -> fp16 [frame, c, size, size].
__global__ void vertical(const FrameView* frames, int max_height, int size, const float* scratch,
                         __half* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int fc = blockIdx.z;
    if (x >= size) return;
    const int c = fc % 3;
    const Taps t = taps_for(y, frames[fc / 3].height, size);
    const float* column = scratch + (std::size_t(fc) * max_height + t.first) * size + x;
    float sum = 0.0f, total = 0.0f;
    for (int j = 0; j < t.count; ++j) {
        const float w = tap_weight(t, j);
        sum += w * column[std::size_t(j) * size];
        total += w;
    }
    const float value = (sum / total / 255.0f - kMean[c]) / kStd[c];
    out[(std::size_t(fc) * size + y) * size + x] = __float2half(value);
}

// One block per (window, frame); one thread per channel.
__global__ void pool_project_kernel(const __half* features, int channels, int side, int window,
                                    int stride, int steps, const float* weight, const float* bias,
                                    int dim, __half* out) {
    extern __shared__ float shared[];
    float* pooled = shared;             // [channels]
    float* projected = shared + channels;  // [dim]
    const int view = blockIdx.x;
    const int frame = blockIdx.y;
    const int top = (view / steps) * stride;
    const int left = (view % steps) * stride;
    const float area = float(window * window);

    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        const __half* map = features + (std::size_t(frame) * channels + c) * side * side;
        float sum = 0.0f;
        for (int y = top; y < top + window; ++y)
            for (int x = left; x < left + window; ++x) sum += __half2float(map[y * side + x]);
        pooled[c] = sum / area;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float* w = weight + std::size_t(d) * channels;
        float sum = bias[d];
        for (int c = 0; c < channels; ++c) sum += w[c] * pooled[c];
        projected[d] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float norm = 0.0f;
        for (int d = 0; d < dim; ++d) norm += projected[d] * projected[d];
        pooled[0] = rsqrtf(fmaxf(norm, 1e-24f));
    }
    __syncthreads();
    const float inv = pooled[0];
    __half* row = out + (std::size_t(frame) * steps * steps + view) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) row[d] = __float2half(projected[d] * inv);
}

}  // namespace

void resize_normalize(const FrameView* frames, int count, int max_height, int size, float* scratch,
                      __half* out, hipStream_t stream) {
    constexpr int threads = 256;
    const dim3 h_grid((size + threads - 1) / threads, max_height, count * 3);
    horizontal<<<h_grid, threads, 0, stream>>>(frames, max_height, size, scratch);
    const dim3 v_grid((size + threads - 1) / threads, size, count * 3);
    vertical<<<v_grid, threads, 0, stream>>>(frames, max_height, size, scratch, out);
}

void pool_project(const __half* features, int count, int channels, int side, int window,
                  int stride, int steps, const float* weight, const float* bias, int dim,
                  __half* out, hipStream_t stream) {
    const dim3 grid(steps * steps, count);
    const std::size_t shared = sizeof(float) * (channels + dim);
    pool_project_kernel<<<grid, 256, shared, stream>>>(features, channels, side, window, stride,
                                                       steps, weight, bias, dim, out);
}

}  // namespace rcm
