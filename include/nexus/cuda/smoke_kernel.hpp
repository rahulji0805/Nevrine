#pragma once

#include <cstddef>

namespace nexus {
namespace cuda {

/**
 * @brief Launches a verified smoke test kernel to confirm GPU execution.
 *
 * Computes: y[i] = alpha * x[i] + beta for i in [0, n)
 *
 * @param d_x Device pointer to input array
 * @param d_y Device pointer to output array
 * @param alpha Scalar multiplier
 * @param beta Scalar bias
 * @param n Number of elements
 */
void launchSmokeKernel(const float* d_x, float* d_y, float alpha, float beta, size_t n);

/**
 * @brief Synchronously runs the full smoke test pipeline:
 *        allocates device buffers, transfers test data, launches kernel,
 *        copies back results, and verifies correctness against CPU reference.
 *
 * @param n Number of elements to test (default 1024)
 * @return true if all elements match CPU reference within 1e-5 tolerance, false otherwise.
 */
bool runSmokeTest(size_t n = 1024);

} // namespace cuda
} // namespace nexus
