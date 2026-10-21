cmake_minimum_required(VERSION 3.25)
# GPU backend: hip (ROCm + MIGraphX; MI325X / MI300X) or cuda (CUDA + TensorRT;
# RTX 40xx). Override the architecture with -DCMAKE_HIP_ARCHITECTURES=... or
# -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs.
set(RCM_BACKEND hip CACHE STRING "GPU backend: hip or cuda")
if(RCM_BACKEND STREQUAL "cuda")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada: RTX 4060
  endif()
  project(rcmindex LANGUAGES CXX CUDA)
elseif(RCM_BACKEND STREQUAL "hip")
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx942)
  endif()
  project(rcmindex LANGUAGES CXX HIP)
else()
  message(FATAL_ERROR "RCM_BACKEND must be hip or cuda, not ${RCM_BACKEND}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CPU image decode (libturbojpeg0-dev, libwebp-dev).
find_library(TURBOJPEG_LIBRARY turbojpeg REQUIRED)
find_path(TURBOJPEG_INCLUDE turbojpeg.h REQUIRED)
find_library(WEBP_LIBRARY webp REQUIRED)
find_path(WEBP_INCLUDE webp/decode.h REQUIRED)

if(RCM_BACKEND STREQUAL "cuda")
  find_package(CUDAToolkit REQUIRED)
  # TensorRT from apt (libnvinfer-dev, libnvonnxparsers-dev) or a tarball: -DTENSORRT_ROOT=...
  find_path(TENSORRT_INCLUDE NvInfer.h HINTS ${TENSORRT_ROOT}/include PATH_SUFFIXES x86_64-linux-gnu REQUIRED)
  find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_ROOT}/lib REQUIRED)
  find_library(NVONNXPARSER_LIBRARY nvonnxparser HINTS ${TENSORRT_ROOT}/lib REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  add_executable(rcmindex src/main.cpp src/kernels.cu)
  target_compile_definitions(rcmindex PRIVATE RCM_CUDA)
  target_compile_options(rcmindex PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  target_include_directories(rcmindex PRIVATE ${TENSORRT_INCLUDE})
  target_link_libraries(rcmindex PRIVATE CUDA::cudart ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY})
else()
  set(CMAKE_HIP_STANDARD 20)
  find_package(hip REQUIRED)
  find_package(migraphx REQUIRED)
  add_executable(rcmindex src/main.cpp src/kernels.hip)
  set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(rcmindex PRIVATE -O3 -Wall -Wextra)
  target_link_libraries(rcmindex PRIVATE hip::host migraphx::c)
endif()

target_include_directories(rcmindex PRIVATE ${TURBOJPEG_INCLUDE} ${WEBP_INCLUDE})
target_link_libraries(rcmindex PRIVATE ${TURBOJPEG_LIBRARY} ${WEBP_LIBRARY})
