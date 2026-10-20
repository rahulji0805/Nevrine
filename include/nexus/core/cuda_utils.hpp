#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nexus {
namespace core {

/**
 * @brief Exception thrown when a CUDA runtime API call fails.
 */
class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t error, const char* file, int line, const char* func, const char* expr)
        : std::runtime_error(formatMessage(error, file, line, func, expr)),
          error_(error),
          line_(line),
          file_(file),
          func_(func),
          expr_(expr) {}

    cudaError_t error() const noexcept { return error_; }
    int line() const noexcept { return line_; }
    const char* file() const noexcept { return file_; }
    const char* func() const noexcept { return func_; }
    const char* expr() const noexcept { return expr_; }

private:
    static std::string formatMessage(cudaError_t err, const char* file, int line,
                                     const char* func, const char* expr) {
        std::ostringstream oss;
        oss << "[CUDA ERROR] " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err)
            << "\n  Failed expression: " << expr
            << "\n  Location: " << file << ":" << line
            << " in function " << func;
        return oss.str();
    }

    cudaError_t error_;
    int line_;
    const char* file_;
    const char* func_;
    const char* expr_;
};

/**
 * @brief Helper function to check CUDA API return codes.
 */
inline void checkCudaError(cudaError_t err, const char* file, int line, const char* func, const char* expr) {
    if (err != cudaSuccess) {
        throw CudaException(err, file, line, func, expr);
    }
}

/**
 * @brief Helper function to check the last executed CUDA kernel error.
 */
inline void checkCudaKernel(const char* file, int line, const char* func, const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "Kernel launch failed: " << kernel_name;
        throw CudaException(err, file, line, func, oss.str().c_str());
    }
}

} // namespace core
} // namespace nexus

#define CUDA_CHECK(call) \
    ::nexus::core::checkCudaError((call), __FILE__, __LINE__, __func__, #call)

#define CUDA_CHECK_KERNEL(kernel_name) \
    ::nexus::core::checkCudaKernel(__FILE__, __LINE__, __func__, (kernel_name))

#define CUDA_SYNC_AND_CHECK() \
    do { \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK_KERNEL("cudaDeviceSynchronize"); \
    } while (0)
