#include "nevrine/cpu/activations_cpu.hpp"
#include <algorithm>
#include <stdexcept>

namespace nevrine {
namespace cpu {

void reluCpuReference(float* data, size_t count) {
    if (data == nullptr && count > 0) {
        throw std::invalid_argument("ReLU data pointer must not be null for non-zero count");
    }

    for (size_t i = 0; i < count; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void leakyReluCpuReference(float* data, size_t count, float negative_slope) {
    if (data == nullptr && count > 0) {
        throw std::invalid_argument("Leaky ReLU data pointer must not be null for non-zero count");
    }

    for (size_t i = 0; i < count; ++i) {
        data[i] = data[i] > 0.0f ? data[i] : negative_slope * data[i];
    }
}

} // namespace cpu
} // namespace nevrine
