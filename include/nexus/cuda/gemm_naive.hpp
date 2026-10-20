#pragma once

#include "nexus/core/cuda_utils.hpp"
#include <cuda_runtime.h>

namespace nexus {
namespace cuda {

/**
 * @brief Launches naive single-precision General Matrix Multiplication (SGEMM) on the GPU.
 *
 * Computes: C = alpha * (A * B) + beta * C
 * Layout: Row-Major
 * Dimensions:
 *   - A: M x K
 *   - B: K x N
 *   - C: M x N
 *
 * Each thread computes one output element (row, col) via naive global memory accesses.
 * Serves as the unoptimized baseline for Week 2 / Week 3 optimization studies.
 *
 * @param A Device pointer to M x K matrix
 * @param B Device pointer to K x N matrix
 * @param C Device pointer to M x N matrix
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A * B (default 1.0f)
 * @param beta Scalar multiplier for C (default 0.0f)
 * @param stream CUDA stream for asynchronous launch (default 0)
 */
void launchNaiveGemm(const float* A, const float* B, float* C,
                    int M, int N, int K,
                    float alpha = 1.0f, float beta = 0.0f,
                    cudaStream_t stream = 0);

} // namespace cuda
} // namespace nexus
