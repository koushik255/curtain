#pragma once

#include <cstdint>

#include "gpu.hpp"

namespace rcm {

// One decoded RGB frame in device memory: channel c of pixel (x, y) is at
// plane[c][y * pitch[c] + x * step] (step 3 for interleaved, 1 for planar).
struct FrameView {
    const std::uint8_t* plane[3];
    std::uint32_t pitch[3];
    std::uint32_t step;
    std::uint32_t width;
    std::uint32_t height;
};

// Antialiased bilinear resize (torch `antialias=True` semantics) of every frame
// to size x size, fused with ImageNet normalization; writes fp16 NCHW.
// `scratch` holds frames * 3 * max_height * size floats.
void resize_normalize(const FrameView* frames, int count, int max_height, int size,
                      float* scratch, __half* out, hipStream_t stream);

// Average-pool `steps` x `steps` windows of `window` cells (stride `stride`) out
// of fp16 [count, channels, side, side] features, project through fc, L2
// normalize, and write fp16 [count, steps*steps, dim].
void pool_project(const __half* features, int count, int channels, int side, int window,
                  int stride, int steps, const float* weight, const float* bias, int dim,
                  __half* out, hipStream_t stream);

}  // namespace rcm
