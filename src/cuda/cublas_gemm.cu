#include "nevrine/cuda/cublas_gemm.hpp"
#include <dlfcn.h>
#include <stdexcept>
#include <utility>

namespace nevrine {
namespace cuda {

CublasHandle::CublasHandle() {
    // In WSL2 environments, ensure the driver library in /usr/lib/wsl/lib is globally visible
    void* d1 = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_GLOBAL | RTLD_NOW);
    if (!d1) {
        std::cerr << "[WARN] dlopen /usr/lib/wsl/lib/libcuda.so.1 failed: " << dlerror() << "\n";
    }

    CUBLAS_CHECK(cublasCreate(&handle_));
}

CublasHandle::~CublasHandle() {
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}

CublasHandle::CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

CublasHandle& CublasHandle::operator=(CublasHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void launchCublasGemm(cublasHandle_t handle,
                      const float* A, const float* B, float* C,
                      int M, int N, int K,
                      float alpha, float beta,
                      cudaStream_t stream) {
    if (handle == nullptr) {
        throw std::invalid_argument("cuBLAS handle must not be null");
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("GEMM matrix dimensions M, N, K must be strictly positive");
    }
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("GEMM pointers A, B, C must not be null");
    }

    if (stream != 0) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    // Row-Major GEMM via Column-Major cuBLAS:
    // C (M x N) = A (M x K) * B (K x N)
    // <=> C^T (N x M) = B^T (N x K) * A^T (K x M)
    // Passing B with leading dimension N, A with leading dimension K, and C with leading dimension N.
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}

} // namespace cuda
} // namespace nevrine
