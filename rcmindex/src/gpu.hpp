#pragma once

// GPU runtime for both backends. The code is written against the HIP runtime
// API; with RCM_CUDA (the CUDA + TensorRT build) the same names map to CUDA,
// which HIP mirrors call for call.

#if defined(RCM_CUDA)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipStream_t cudaStream_t
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamDestroy cudaStreamDestroy
#define hipEvent_t cudaEvent_t
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipHostMalloc cudaMallocHost
#define hipHostFree cudaFreeHost
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties

#else

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#endif
