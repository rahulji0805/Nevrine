#include "nevrine/cuda/gemm_naive.hpp"
#include <stdexcept>

namespace nevrine {
namespace cuda {

__global__ void naiveGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        if (beta == 0.0f) {
            C[row * N + col] = alpha * sum;
        } else {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

void launchNaiveGemm(const float* A, const float* B, float* C,
                    int M, int N, int K,
                    float alpha, float beta,
                    cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("GEMM matrix dimensions M, N, K must be strictly positive");
    }
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("GEMM pointers A, B, C must not be null");
    }

    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (M + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    naiveGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_KERNEL("naiveGemmKernel");
}

} // namespace cuda
} // namespace nevrine
