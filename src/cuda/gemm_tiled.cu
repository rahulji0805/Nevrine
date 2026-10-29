#include "nevrine/cuda/gemm_tiled.hpp"
#include <stdexcept>

namespace nevrine {
namespace cuda {

namespace {

constexpr int TILE_DIM = 32;

__global__ void tiledGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta) {
    // 2D shared memory cache with +1 column padding to prevent bank conflicts
    __shared__ float s_A[TILE_DIM][TILE_DIM + 1];
    __shared__ float s_B[TILE_DIM][TILE_DIM + 1];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // Collaborative loading of tile from Matrix A (row-major: M x K)
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Collaborative loading of tile from Matrix B (row-major: K x N)
        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Wait for all threads in threadblock to finish loading shared memory
        __syncthreads();

        // Accumulate partial dot product across the current tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Wait for all threads to finish computing before loading next tile
        __syncthreads();
    }

    // Write final accumulated result to Matrix C (row-major: M x N)
    if (row < M && col < N) {
        int idx = row * N + col;
        if (beta == 0.0f) {
            C[idx] = alpha * acc;
        } else {
            C[idx] = alpha * acc + beta * C[idx];
        }
    }
}

} // anonymous namespace

void launchTiledGemm(const float* A, const float* B, float* C,
                    int M, int N, int K,
                    float alpha, float beta,
                    cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("GEMM matrix dimensions M, N, K must be strictly positive");
    }
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("GEMM pointers A, B, C must not be null");
    }

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM);

    tiledGemmKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_KERNEL("tiledGemmKernel");
}

} // namespace cuda
} // namespace nevrine
