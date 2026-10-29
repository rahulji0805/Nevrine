#include "nevrine/core/arena.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include "nevrine/cpu/gemm_cpu.hpp"
#include "nevrine/cuda/cublas_gemm.hpp"
#include "nevrine/cuda/gemm_tiled.hpp"
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

void runSingleTiledGemmTest(nevrine::cuda::CublasHandle& cublas,
                           int M, int N, int K,
                           float alpha, float beta,
                           const std::string& label) {
    std::cout << "  Testing " << label << " (M=" << M << ", N=" << N << ", K=" << K
              << ", alpha=" << alpha << ", beta=" << beta << ")... " << std::flush;

    const size_t size_A = M * K;
    const size_t size_B = K * N;
    const size_t size_C = M * N;

    const size_t bytes_A = size_A * sizeof(float);
    const size_t bytes_B = size_B * sizeof(float);
    const size_t bytes_C = size_C * sizeof(float);

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_init(size_C, 0.0f);
    std::vector<float> h_C_cpu(size_C, 0.0f);
    std::vector<float> h_C_tiled(size_C, 0.0f);
    std::vector<float> h_C_cublas(size_C, 0.0f);

    fillRandom(h_A);
    fillRandom(h_B);

    if (beta != 0.0f) {
        fillRandom(h_C_init);
        h_C_cpu = h_C_init;
    }

    // 1. CPU Golden Reference
    nevrine::cpu::gemmCpuReference(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K, alpha, beta);

    // 2. Allocate GPU memory via DeviceMemoryArena
    size_t arena_capacity = bytes_A + bytes_B + bytes_C * 2 + 8192;
    nevrine::core::DeviceMemoryArena arena(arena_capacity, 256);

    float* d_A = static_cast<float*>(arena.allocate(bytes_A));
    float* d_B = static_cast<float*>(arena.allocate(bytes_B));
    float* d_C_tiled = static_cast<float*>(arena.allocate(bytes_C));
    float* d_C_cublas = static_cast<float*>(arena.allocate(bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    if (beta != 0.0f) {
        CUDA_CHECK(cudaMemcpy(d_C_tiled, h_C_init.data(), bytes_C, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C_init.data(), bytes_C, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_C_tiled, 0, bytes_C));
        CUDA_CHECK(cudaMemset(d_C_cublas, 0, bytes_C));
    }

    // 3. Run Tiled GEMM kernel
    nevrine::cuda::launchTiledGemm(d_A, d_B, d_C_tiled, M, N, K, alpha, beta);
    CUDA_SYNC_AND_CHECK();

    // 4. Run cuBLAS GEMM reference
    nevrine::cuda::launchCublasGemm(cublas.get(), d_A, d_B, d_C_cublas, M, N, K, alpha, beta);
    CUDA_SYNC_AND_CHECK();

    // 5. Transfer back to host
    CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C_tiled, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, bytes_C, cudaMemcpyDeviceToHost));

    // 6. Verification: Tiled vs CPU and Tiled vs cuBLAS
    float max_diff_cpu = 0.0f;
    float max_diff_cublas = 0.0f;

    for (size_t i = 0; i < size_C; ++i) {
        float diff_cpu = std::fabs(h_C_tiled[i] - h_C_cpu[i]);
        float diff_cublas = std::fabs(h_C_tiled[i] - h_C_cublas[i]);

        if (diff_cpu > max_diff_cpu) max_diff_cpu = diff_cpu;
        if (diff_cublas > max_diff_cublas) max_diff_cublas = diff_cublas;

        if (diff_cpu > 1e-3f) {
            std::cerr << "FAILED! Mismatch vs CPU at element " << i
                      << ": Tiled=" << h_C_tiled[i] << ", CPU=" << h_C_cpu[i]
                      << ", diff=" << diff_cpu << "\n";
            throw std::runtime_error("Tiled GEMM numerical tolerance vs CPU exceeded");
        }
        if (diff_cublas > 1e-3f) {
            std::cerr << "FAILED! Mismatch vs cuBLAS at element " << i
                      << ": Tiled=" << h_C_tiled[i] << ", cuBLAS=" << h_C_cublas[i]
                      << ", diff=" << diff_cublas << "\n";
            throw std::runtime_error("Tiled GEMM numerical tolerance vs cuBLAS exceeded");
        }
    }

    std::cout << "PASSED (max_diff_cpu=" << max_diff_cpu
              << ", max_diff_cublas=" << max_diff_cublas << ")\n";
}

int main() {
    std::cout << "========================================================\n"
              << "       Nevrine TILED GEMM & cuBLAS TEST SUITE (P3)    \n"
              << "========================================================\n";

    try {
        nevrine::cuda::CublasHandle cublas;

        // 1. Square matrix powers of two
        runSingleTiledGemmTest(cublas, 32, 32, 32, 1.0f, 0.0f, "Small Square (32x32)");
        runSingleTiledGemmTest(cublas, 128, 128, 128, 1.0f, 0.0f, "Medium Square (128x128)");
        runSingleTiledGemmTest(cublas, 512, 512, 512, 1.0f, 0.0f, "Standard Square (512x512)");

        // 2. Non-multiples of tile dimension (16) and prime dimensions
        runSingleTiledGemmTest(cublas, 17, 31, 43, 1.0f, 0.0f, "Prime Dimensions (17x31x43)");
        runSingleTiledGemmTest(cublas, 7, 13, 19, 1.0f, 0.0f, "Small Irregular (7x13x19)");
        runSingleTiledGemmTest(cublas, 65, 33, 49, 1.0f, 0.0f, "Tile+1 Edge Case (65x33x49)");

        // 3. Rectangular shapes (tall and wide)
        runSingleTiledGemmTest(cublas, 64, 256, 128, 1.0f, 0.0f, "Wide Matrix (64x256x128)");
        runSingleTiledGemmTest(cublas, 256, 64, 128, 1.0f, 0.0f, "Tall Matrix (256x64x128)");

        // 4. Alpha & Beta scaling
        runSingleTiledGemmTest(cublas, 64, 64, 64, 2.5f, 1.5f, "Scaled GEMM (alpha=2.5, beta=1.5)");

        // 5. Invalid arguments check
        std::cout << "  Testing Invalid Dimensions Exception Handling... " << std::flush;
        bool caught = false;
        try {
            nevrine::cuda::launchTiledGemm(nullptr, nullptr, nullptr, 0, 10, 10);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) throw std::runtime_error("Failed to catch invalid dimensions");
        std::cout << "PASSED\n";

        std::cout << "\n[ALL TESTS PASSED] Tiled SGEMM matches CPU reference and cuBLAS across all test cases!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED] " << e.what() << "\n";
        return 1;
    }
}
