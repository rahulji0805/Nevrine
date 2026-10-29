#pragma once

#include "nevrine/core/cuda_utils.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace nevrine {
namespace cuda {

/**
 * @brief RAII wrapper for cuBLAS library handle.
 */
class CublasHandle {
public:
    CublasHandle();
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept;
    CublasHandle& operator=(CublasHandle&& other) noexcept;

    cublasHandle_t get() const noexcept { return handle_; }
    operator cublasHandle_t() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

/**
 * @brief Performs single-precision General Matrix Multiplication (SGEMM) via cuBLAS in Row-Major order.
 *
 * Computes: C = alpha * (A * B) + beta * C
 * Layout: Row-Major
 * Dimensions:
 *   - A: M x K
 *   - B: K x N
 *   - C: M x N
 *
 * Implements the standard row-major algebraic mapping C = A * B <=> C^T = B^T * A^T
 * on cuBLAS's native column-major interface without requiring physical memory transposition.
 *
 * @param handle Valid cuBLAS library handle
 * @param A Device pointer to M x K matrix
 * @param B Device pointer to K x N matrix
 * @param C Device pointer to M x N matrix
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A * B (default 1.0f)
 * @param beta Scalar multiplier for C (default 0.0f)
 * @param stream CUDA stream for execution (default 0)
 */
void launchCublasGemm(cublasHandle_t handle,
                      const float* A, const float* B, float* C,
                      int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f,
                      cudaStream_t stream = 0);

} // namespace cuda
} // namespace nevrine
