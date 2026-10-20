cmake_minimum_required(VERSION 3.20)
project(nexus_cuda LANGUAGES CXX CUDA)

# C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard & Architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Target architectures: sm_89 (RTX 4060 Ada) and sm_75 (AWS T4 Turing)
set(CMAKE_CUDA_ARCHITECTURES "75;89")

message(STATUS "NEXUS-CUDA build configuration:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    # nvcc host compiler options and warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-pedantic>)
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Core library sources
add_library(nexus_core STATIC
    src/core/device_info.cpp
    src/core/arena.cpp
    src/cpu/gemm_cpu.cpp
    src/cuda/smoke_kernel.cu
    src/cuda/gemm_naive.cu
)

target_include_directories(nexus_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Environment verification & smoke test executable (Session 1 / P0)
add_executable(nexus_smoke_test
    src/main.cpp
)
target_link_libraries(nexus_smoke_test PRIVATE nexus_core)

# Memory arena unit test executable (Session 2 / P1)
add_executable(test_arena
    tests/cpp/test_arena.cu
)
target_link_libraries(test_arena PRIVATE nexus_core)

# Naive GEMM unit & reference test executable (Session 3 / P2)
add_executable(test_gemm
    tests/cpp/test_gemm.cu
)
target_link_libraries(test_gemm PRIVATE nexus_core)

# Naive GEMM baseline benchmark harness (Session 3 / P2)
add_executable(bench_gemm_naive
    benchmarks/bench_gemm_naive.cu
)
target_link_libraries(bench_gemm_naive PRIVATE nexus_core)

# Enable testing support
enable_testing()
add_test(NAME smoke_test COMMAND nexus_smoke_test)
add_test(NAME arena_test COMMAND test_arena)
add_test(NAME gemm_naive_test COMMAND test_gemm)
