#include "nexus/cpu/gemm_cpu.hpp"
#include <stdexcept>

namespace nexus {
namespace cpu {

void gemmCpuReference(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      float alpha, float beta) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("GEMM matrix dimensions M, N, K must be strictly positive");
    }
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("GEMM pointers A, B, C must not be null");
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }

            if (beta == 0.0f) {
                C[i * N + j] = alpha * sum;
            } else {
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    }
}

} // namespace cpu
} // namespace nexus
