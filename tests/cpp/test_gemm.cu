#include "nexus/core/arena.hpp"
#include "nexus/core/cuda_utils.hpp"
#include "nexus/cpu/gemm_cpu.hpp"
#include "nexus/cuda/gemm_naive.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void fillRandom(std::vector<float>& vec, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& val : vec) {
        val = dist(rng);
    }
}

void runSingleGemmTest(int M, int N, int K, float alpha, float beta, const std::string& label) {
    std::cout << "  Testing " << label << " (M=" << M << ", N=" << N << ", K=" << K
              << ", alpha=" << alpha << ", beta=" << beta << ")... ";

    const size_t size_A = M * K;
    const size_t size_B = K * N;
    const size_t size_C = M * N;

    const size_t bytes_A = size_A * sizeof(float);
    const size_t bytes_B = size_B * sizeof(float);
    const size_t bytes_C = size_C * sizeof(float);

    // Host memory
    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_cpu(size_C, 0.0f);
    std::vector<float> h_C_gpu(size_C, 0.0f);

    fillRandom(h_A);
    fillRandom(h_B);

    if (beta != 0.0f) {
        fillRandom(h_C_cpu);
        h_C_gpu = h_C_cpu;
    }

    // Run CPU Golden Reference
    nexus::cpu::gemmCpuReference(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K, alpha, beta);

    // Run GPU Naive GEMM via DeviceMemoryArena
    size_t arena_capacity = bytes_A + bytes_B + bytes_C + 4096; // slight headroom for 256-byte alignment
    nexus::core::DeviceMemoryArena arena(arena_capacity, 256);

    float* d_A = static_cast<float*>(arena.allocate(bytes_A));
    float* d_B = static_cast<float*>(arena.allocate(bytes_B));
    float* d_C = static_cast<float*>(arena.allocate(bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    if (beta != 0.0f) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C_gpu.data(), bytes_C, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    }

    nexus::cuda::launchNaiveGemm(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_SYNC_AND_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Compare results
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_C; ++i) {
        float diff = std::fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 1e-3f) {
            std::cerr << "FAILED! Mismatch at element " << i
                      << ": GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i]
                      << ", diff=" << diff << "\n";
            throw std::runtime_error("GEMM numerical tolerance exceeded");
        }
    }

    std::cout << "PASSED (max_diff=" << max_diff << ")\n";
}

int main() {
    std::cout << "========================================================\n"
              << "         NEXUS-CUDA NAIVE GEMM TEST SUITE (P2)          \n"
              << "========================================================\n";

    try {
        // 1. Small square matrix (powers of 2)
        runSingleGemmTest(32, 32, 32, 1.0f, 0.0f, "Small Square (32x32)");
        runSingleGemmTest(128, 128, 128, 1.0f, 0.0f, "Medium Square (128x128)");

        // 2. Non-multiples of block size (edge dimensions & odd shapes)
        runSingleGemmTest(17, 31, 43, 1.0f, 0.0f, "Prime Dimensions (17x31x43)");
        runSingleGemmTest(7, 13, 19, 1.0f, 0.0f, "Small Irregular (7x13x19)");

        // 3. Rectangular (tall and wide)
        runSingleGemmTest(64, 256, 128, 1.0f, 0.0f, "Wide Matrix (64x256x128)");
        runSingleGemmTest(256, 64, 128, 1.0f, 0.0f, "Tall Matrix (256x64x128)");

        // 4. Alpha & Beta scaling
        runSingleGemmTest(64, 64, 64, 2.5f, 1.5f, "Scaled GEMM (alpha=2.5, beta=1.5)");

        std::cout << "\n[ALL TESTS PASSED] Naive SGEMM baseline matches CPU reference across all shapes!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED] " << e.what() << "\n";
        return 1;
    }
}
