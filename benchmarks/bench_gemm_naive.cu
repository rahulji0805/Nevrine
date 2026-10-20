#include "nevrine/core/arena.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include "nevrine/cuda/gemm_naive.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

struct BenchmarkResult {
    int M;
    int N;
    int K;
    float median_ms;
    float p95_ms;
    double gflops;
};

BenchmarkResult runGemmBenchmark(int M, int N, int K, int warmup = 5, int iters = 20) {
    const size_t size_A = M * K;
    const size_t size_B = K * N;
    const size_t size_C = M * N;

    const size_t bytes_A = size_A * sizeof(float);
    const size_t bytes_B = size_B * sizeof(float);
    const size_t bytes_C = size_C * sizeof(float);

    const size_t arena_capacity = bytes_A + bytes_B + bytes_C + 4096;
    nevrine::core::DeviceMemoryArena arena(arena_capacity, 256);

    float* d_A = static_cast<float*>(arena.allocate(bytes_A));
    float* d_B = static_cast<float*>(arena.allocate(bytes_B));
    float* d_C = static_cast<float*>(arena.allocate(bytes_C));

    CUDA_CHECK(cudaMemset(d_A, 0x3F, bytes_A)); // fill with ~0.75f
    CUDA_CHECK(cudaMemset(d_B, 0x3F, bytes_B));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    // Warm-up runs
    for (int i = 0; i < warmup; ++i) {
        nevrine::cuda::launchNaiveGemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    }
    CUDA_SYNC_AND_CHECK();

    // Measurement runs using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> latencies_ms(iters);

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        nevrine::cuda::launchNaiveGemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
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

    // Compute GFLOP/s: (2 * M * N * K) / (time_in_seconds * 1e9)
    double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gflops = (total_flops / (static_cast<double>(median) * 1e-3)) * 1e-9;

    return {M, N, K, median, p95, gflops};
}

int main() {
    std::cout << "=========================================================================\n"
              << "              Nevrine NAIVE SGEMM BASELINE BENCHMARK                  \n"
              << "=========================================================================\n"
              << "Device: NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace sm_89)\n"
              << "Timing: CUDA Events | Warm-up: 5 iterations | Measurement: 20 iterations\n"
              << "-------------------------------------------------------------------------\n";

    std::cout << std::left
              << std::setw(18) << "Shape (MxNxK)"
              << std::setw(16) << "Median (ms)"
              << std::setw(16) << "p95 (ms)"
              << std::setw(18) << "Throughput (GFLOP/s)"
              << "\n"
              << "-------------------------------------------------------------------------\n";

    const std::vector<int> sizes = {256, 512, 1024, 2048};

    for (int sz : sizes) {
        BenchmarkResult res = runGemmBenchmark(sz, sz, sz);
        std::string shape_str = std::to_string(sz) + "x" + std::to_string(sz) + "x" + std::to_string(sz);
        std::cout << std::left
                  << std::setw(18) << shape_str
                  << std::setw(16) << std::fixed << std::setprecision(4) << res.median_ms
                  << std::setw(16) << std::fixed << std::setprecision(4) << res.p95_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << res.gflops
                  << "\n";
    }

    std::cout << "=========================================================================\n";
    std::cout << "Baseline captured! Next step: Week 3 Tiled GEMM with shared memory.\n";

    return 0;
}
