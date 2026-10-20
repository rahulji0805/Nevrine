#include "nevrine/cuda/smoke_kernel.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace nevrine {
namespace cuda {

// Device kernel for elementwise affine transform: y[i] = alpha * x[i] + beta
__global__ void smokeKernel(const float* __restrict__ x, float* __restrict__ y, float alpha, float beta, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + beta;
    }
}

void launchSmokeKernel(const float* d_x, float* d_y, float alpha, float beta, size_t n) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    smokeKernel<<<grid_size, block_size>>>(d_x, d_y, alpha, beta, n);
    CUDA_CHECK_KERNEL("smokeKernel");
}

bool runSmokeTest(size_t n) {
    const float alpha = 3.14159f;
    const float beta = 2.71828f;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_y(n, 0.0f);
    std::vector<float> h_expected(n);

    for (size_t i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i % 100) * 0.1f;
        h_expected[i] = alpha * h_x[i] + beta;
    }

    float* d_x = nullptr;
    float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, bytes));

    launchSmokeKernel(d_x, d_y, alpha, beta, n);
    CUDA_SYNC_AND_CHECK();

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Verify results
    size_t mismatches = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(h_y[i] - h_expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 1e-4f) {
            if (mismatches < 5) {
                std::cerr << "Mismatch at index " << i << ": expected " << h_expected[i]
                          << ", got " << h_y[i] << " (diff=" << diff << ")\n";
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "[PASS] Smoke Kernel verified on " << n << " elements. Max absolute error: "
                  << max_diff << "\n";
        return true;
    } else {
        std::cerr << "[FAIL] Smoke Kernel verification failed with " << mismatches
                  << " mismatches!\n";
        return false;
    }
}

} // namespace cuda
} // namespace nevrine
