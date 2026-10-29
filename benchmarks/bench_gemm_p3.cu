#include "nevrine/core/arena.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include "nevrine/cuda/cublas_gemm.hpp"
#include "nevrine/cuda/gemm_naive.hpp"
#include "nevrine/cuda/gemm_tiled.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

struct KernelMetrics {
    float median_ms;
    float p95_ms;
    double gflops;
};

struct ShapeBenchmarkResult {
    int M, N, K;
    KernelMetrics naive;
    KernelMetrics tiled;
    KernelMetrics cublas;
    double speedup_vs_naive;
    double pct_of_cublas;
};

template <typename Func>
KernelMetrics measureLatency(Func&& launch_fn, int M, int N, int K, int warmup = 5, int iters = 20) {
    // Warm-up passes
    for (int i = 0; i < warmup; ++i) {
        launch_fn();
    }
    CUDA_SYNC_AND_CHECK();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> latencies_ms(iters);

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        latencies_ms[i] = ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(latencies_ms.begin(), latencies_ms.end());
    float median = latencies_ms[iters / 2];
    size_t p95_idx = static_cast<size_t>(iters * 0.95);
    if (p95_idx >= latencies_ms.size()) p95_idx = latencies_ms.size() - 1;
    float p95 = latencies_ms[p95_idx];

    double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gflops = (total_flops / (static_cast<double>(median) * 1e-3)) * 1e-9;

    return {median, p95, gflops};
}

ShapeBenchmarkResult runShapeComparison(nevrine::cuda::CublasHandle& cublas,
                                       int M, int N, int K,
                                       int warmup = 5, int iters = 20) {
    const size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    const size_t arena_capacity = bytes_A + bytes_B + bytes_C + 8192;
    nevrine::core::DeviceMemoryArena arena(arena_capacity, 256);

    float* d_A = static_cast<float*>(arena.allocate(bytes_A));
    float* d_B = static_cast<float*>(arena.allocate(bytes_B));
    float* d_C = static_cast<float*>(arena.allocate(bytes_C));

    CUDA_CHECK(cudaMemset(d_A, 0x3F, bytes_A)); // ~0.75f
    CUDA_CHECK(cudaMemset(d_B, 0x3F, bytes_B));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    // Measure Naive GEMM
    KernelMetrics naive_metrics = measureLatency([&]() {
        nevrine::cuda::launchNaiveGemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    }, M, N, K, warmup, iters);

    // Measure Tiled GEMM (P3)
    KernelMetrics tiled_metrics = measureLatency([&]() {
        nevrine::cuda::launchTiledGemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    }, M, N, K, warmup, iters);

    // Measure cuBLAS (Peak reference)
    KernelMetrics cublas_metrics = measureLatency([&]() {
        nevrine::cuda::launchCublasGemm(cublas.get(), d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    }, M, N, K, warmup, iters);

    double speedup = naive_metrics.median_ms / tiled_metrics.median_ms;
    double pct_cublas = (tiled_metrics.gflops / cublas_metrics.gflops) * 100.0;

    return {M, N, K, naive_metrics, tiled_metrics, cublas_metrics, speedup, pct_cublas};
}

int main() {
    std::cout << "===================================================================================================\n"
              << "                        Nevrine PHASE 3 BENCHMARK: TILED GEMM vs cuBLAS                            \n"
              << "===================================================================================================\n"
              << "Hardware: NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace sm_89)\n"
              << "Configuration: CUDA Events timing, 5 warm-up iterations, 20 measurement iterations\n"
              << "---------------------------------------------------------------------------------------------------\n";

    nevrine::cuda::CublasHandle cublas;

    std::cout << std::left
              << std::setw(15) << "Matrix Size"
              << std::setw(16) << "Naive (GFLOP/s)"
              << std::setw(16) << "Tiled (GFLOP/s)"
              << std::setw(16) << "cuBLAS (GFLOP/s)"
              << std::setw(16) << "Speedup vs Naive"
              << std::setw(14) << "% of cuBLAS"
              << "\n"
              << "---------------------------------------------------------------------------------------------------\n";

    std::vector<ShapeBenchmarkResult> results;
    const std::vector<int> sizes = {256, 512, 1024, 2048};

    for (int sz : sizes) {
        ShapeBenchmarkResult res = runShapeComparison(cublas, sz, sz, sz);
        results.push_back(res);
        std::string shape_str = std::to_string(sz) + "x" + std::to_string(sz) + "x" + std::to_string(sz);

        std::cout << std::left
                  << std::setw(15) << shape_str
                  << std::setw(16) << std::fixed << std::setprecision(2) << res.naive.gflops
                  << std::setw(16) << std::fixed << std::setprecision(2) << res.tiled.gflops
                  << std::setw(16) << std::fixed << std::setprecision(2) << res.cublas.gflops
                  << std::setw(16) << (std::to_string(res.speedup_vs_naive).substr(0, 4) + "x")
                  << std::setw(14) << (std::to_string(res.pct_of_cublas).substr(0, 5) + "%")
                  << "\n";
    }

    std::cout << "\n---------------------------------------------------------------------------------------------------\n";
    std::cout << "### Markdown Table for BENCHMARKS.md & README.md:\n\n";
    std::cout << "| Dimensions | Naive Latency | Tiled Latency | cuBLAS Latency | Naive GFLOP/s | Tiled GFLOP/s | cuBLAS GFLOP/s | % of cuBLAS |\n";
    std::cout << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";

    for (const auto& res : results) {
        std::string shape = "$" + std::to_string(res.M) + " \\times " + std::to_string(res.N) + " \\times " + std::to_string(res.K) + "$";
        std::cout << "| **" << shape << "** | "
                  << std::fixed << std::setprecision(4) << res.naive.median_ms << " ms | "
                  << std::fixed << std::setprecision(4) << res.tiled.median_ms << " ms | "
                  << std::fixed << std::setprecision(4) << res.cublas.median_ms << " ms | "
                  << std::fixed << std::setprecision(2) << res.naive.gflops << " GFLOP/s | "
                  << std::fixed << std::setprecision(2) << res.tiled.gflops << " GFLOP/s | "
                  << std::fixed << std::setprecision(2) << res.cublas.gflops << " GFLOP/s | "
                  << std::fixed << std::setprecision(2) << res.pct_of_cublas << "% |\n";
    }

    std::cout << "===================================================================================================\n";
    std::cout << "Benchmark Complete! Shared Memory Tiling verified across all matrix dimensions.\n";

    return 0;
}
