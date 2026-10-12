cmake_minimum_required(VERSION 3.25)
# MI325X / MI300X. Override with -DCMAKE_HIP_ARCHITECTURES=... for other GPUs.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx942)
endif()
project(rcmindex LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(migraphx REQUIRED)
find_package(rocjpeg REQUIRED)

add_executable(rcmindex src/main.cpp src/kernels.hip)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(rcmindex PRIVATE -O3 -Wall -Wextra)
target_link_libraries(rcmindex PRIVATE hip::host migraphx::c rocjpeg::rocjpeg)
