#pragma once

namespace nexus {
namespace cpu {

/**
 * @brief Golden CPU reference implementation for SGEMM: C = alpha * (A * B) + beta * C
 * Layout: Row-Major
 *
 * @param A Pointer to M x K matrix on host
 * @param B Pointer to K x N matrix on host
 * @param C Pointer to M x N matrix on host
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A * B
 * @param beta Scalar multiplier for C
 */
void gemmCpuReference(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f);

} // namespace cpu
} // namespace nexus
