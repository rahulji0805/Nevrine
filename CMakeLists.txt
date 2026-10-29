cmake_minimum_required(VERSION 3.20)
project(nevrine LANGUAGES CXX CUDA)

# C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA Standard & Architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Target architectures: sm_89 (RTX 4060 Ada) and sm_75 (AWS T4 Turing)
set(CMAKE_CUDA_ARCHITECTURES "75;89")

# WSL2 driver library RPATH
if(EXISTS "/usr/lib/wsl/lib")
    set(CMAKE_BUILD_RPATH "/usr/lib/wsl/lib")
    set(CMAKE_INSTALL_RPATH "/usr/lib/wsl/lib")
endif()

message(STATUS "Nevrine build configuration:")
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

# Find CUDA Toolkit for cuBLAS
find_package(CUDAToolkit REQUIRED)

# Core library sources
add_library(nevrine_core STATIC
    src/core/device_info.cpp
    src/core/arena.cpp
    src/cpu/gemm_cpu.cpp
    src/cpu/activations_cpu.cpp
    src/cpu/conv2d_cpu.cpp
    src/cuda/smoke_kernel.cu
    src/cuda/gemm_naive.cu
    src/cuda/gemm_tiled.cu
    src/cuda/cublas_gemm.cu
    src/cuda/activations.cu
    src/cuda/conv2d.cu
)

target_include_directories(nevrine_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(nevrine_core PUBLIC
    CUDA::cublas
)

# Environment verification & smoke test executable (Session 1 / P0)
add_executable(nevrine_smoke
    src/main.cpp
)
target_link_libraries(nevrine_smoke PRIVATE nevrine_core)

# Memory arena unit test executable (Session 2 / P1)
add_executable(test_arena
    tests/cpp/test_arena.cu
)
target_link_libraries(test_arena PRIVATE nevrine_core)

# Naive GEMM unit & reference test executable (Session 3 / P2)
add_executable(test_gemm
    tests/cpp/test_gemm.cu
)
target_link_libraries(test_gemm PRIVATE nevrine_core)

# Naive GEMM baseline benchmark harness (Session 3 / P2)
add_executable(bench_gemm_naive
    benchmarks/bench_gemm_naive.cu
)
target_link_libraries(bench_gemm_naive PRIVATE nevrine_core)

# Tiled GEMM & cuBLAS unit test executable (Week 3 / P3)
add_executable(test_gemm_tiled
    tests/cpp/test_gemm_tiled.cu
)
target_link_libraries(test_gemm_tiled PRIVATE nevrine_core)

# P3 Comparative Benchmark: Naive vs Tiled vs cuBLAS (Week 3 / P3)
add_executable(bench_gemm_p3
    benchmarks/bench_gemm_p3.cu
)
target_link_libraries(bench_gemm_p3 PRIVATE nevrine_core)

# Activations (ReLU / Leaky ReLU) unit test executable (Week 4 / P4)
add_executable(test_activations
    tests/cpp/test_activations.cu
)
target_link_libraries(test_activations PRIVATE nevrine_core)

# Conv2D & Fused Conv2D+ReLU unit test executable (Week 4 / P4)
add_executable(test_conv2d
    tests/cpp/test_conv2d.cu
)
target_link_libraries(test_conv2d PRIVATE nevrine_core)

# Python3 & Pybind11 Integration (Week 5 / P5)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_PROC_RESULT
)
if(pybind11_PROC_RESULT EQUAL 0 AND EXISTS "${pybind11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_nevrine
    src/python/bindings.cpp
)

target_link_libraries(_nevrine PRIVATE
    nevrine_core
)

set_target_properties(_nevrine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/nevrine"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/nevrine"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/nevrine"
)

# Enable testing support
enable_testing()
add_test(NAME smoke_test COMMAND nevrine_smoke)
add_test(NAME arena_test COMMAND test_arena)
add_test(NAME gemm_naive_test COMMAND test_gemm)
add_test(NAME gemm_tiled_test COMMAND test_gemm_tiled)
add_test(NAME activations_test COMMAND test_activations)
add_test(NAME conv2d_test COMMAND test_conv2d)
add_test(NAME pytest_suite COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_SOURCE_DIR}/tests/python" -v)

if(EXISTS "/usr/lib/wsl/lib")
    set_tests_properties(smoke_test arena_test gemm_naive_test gemm_tiled_test activations_test conv2d_test pytest_suite PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib:$ENV{LD_LIBRARY_PATH};PYTHONPATH=${CMAKE_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
    )
else()
    set_tests_properties(pytest_suite PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
    )
endif()
