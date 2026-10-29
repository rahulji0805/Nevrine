#include "nevrine/cuda/activations.hpp"
#include <stdexcept>

namespace nevrine {
namespace cuda {

namespace {

constexpr int ACTIVATION_BLOCK_SIZE = 256;

__global__ void reluKernel(float* __restrict__ data, size_t count) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < count) {
        float val = data[idx];
        data[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void leakyReluKernel(float* __restrict__ data, size_t count,
                                 float negative_slope) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < count) {
        float val = data[idx];
        data[idx] = val > 0.0f ? val : negative_slope * val;
    }
}

} // anonymous namespace

void launchRelu(float* data, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    if (data == nullptr) {
        throw std::invalid_argument("ReLU data pointer must not be null");
    }

    int grid = static_cast<int>((count + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE);
    reluKernel<<<grid, ACTIVATION_BLOCK_SIZE, 0, stream>>>(data, count);
    CUDA_CHECK_KERNEL("reluKernel");
}

void launchLeakyRelu(float* data, size_t count,
                     float negative_slope,
                     cudaStream_t stream) {
    if (count == 0) return;
    if (data == nullptr) {
        throw std::invalid_argument("Leaky ReLU data pointer must not be null");
    }

    int grid = static_cast<int>((count + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE);
    leakyReluKernel<<<grid, ACTIVATION_BLOCK_SIZE, 0, stream>>>(data, count, negative_slope);
    CUDA_CHECK_KERNEL("leakyReluKernel");
}

} // namespace cuda
} // namespace nevrine
