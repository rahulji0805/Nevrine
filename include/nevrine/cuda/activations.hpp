#pragma once

#include "nevrine/core/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace nevrine {
namespace cuda {

/**
 * @brief Launches in-place ReLU activation on the GPU.
 *
 * Applies x[i] = max(0.0f, x[i]) for all elements.
 * Uses a 1D grid with 256 threads per block.
 * Boundary-predicated for arbitrary tensor sizes.
 *
 * @param data Device pointer to float array (modified in-place)
 * @param count Number of elements
 * @param stream CUDA stream for asynchronous launch (default 0)
 */
void launchRelu(float* data, size_t count, cudaStream_t stream = 0);

/**
 * @brief Launches in-place Leaky ReLU activation on the GPU.
 *
 * Applies x[i] = x[i] > 0 ? x[i] : negative_slope * x[i] for all elements.
 * Uses a 1D grid with 256 threads per block.
 * Boundary-predicated for arbitrary tensor sizes.
 *
 * @param data Device pointer to float array (modified in-place)
 * @param count Number of elements
 * @param negative_slope Slope for negative values (default 0.01f)
 * @param stream CUDA stream for asynchronous launch (default 0)
 */
void launchLeakyRelu(float* data, size_t count,
                     float negative_slope = 0.01f,
                     cudaStream_t stream = 0);

} // namespace cuda
} // namespace nevrine
