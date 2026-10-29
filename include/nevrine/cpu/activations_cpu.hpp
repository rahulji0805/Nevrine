#pragma once

#include <cstddef>

namespace nevrine {
namespace cpu {

/**
 * @brief CPU golden reference: in-place ReLU activation.
 *
 * Applies x[i] = max(0.0f, x[i]) for all elements.
 *
 * @param data Pointer to host float array
 * @param count Number of elements
 */
void reluCpuReference(float* data, size_t count);

/**
 * @brief CPU golden reference: in-place Leaky ReLU activation.
 *
 * Applies x[i] = x[i] > 0 ? x[i] : negative_slope * x[i] for all elements.
 *
 * @param data Pointer to host float array
 * @param count Number of elements
 * @param negative_slope Slope for negative values (default 0.01f)
 */
void leakyReluCpuReference(float* data, size_t count, float negative_slope = 0.01f);

} // namespace cpu
} // namespace nevrine
