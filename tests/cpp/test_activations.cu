#include "nevrine/core/arena.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include "nevrine/cpu/activations_cpu.hpp"
#include "nevrine/cuda/activations.hpp"
#include <cmath>
#include <iostream>
#include <vector>

/**
 * @brief Verifies GPU activation output against CPU golden reference.
 */
void verifyActivation(const std::vector<float>& gpu_result,
                      const std::vector<float>& cpu_reference,
                      const std::string& label,
                      float tolerance = 1e-6f) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = std::fabs(gpu_result[i] - cpu_reference[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            std::cerr << "FAILED! " << label << " mismatch at element " << i
                      << ": GPU=" << gpu_result[i] << ", CPU=" << cpu_reference[i]
                      << ", diff=" << diff << "\n";
            throw std::runtime_error(label + " numerical tolerance exceeded");
        }
    }
    std::cout << "PASSED (max_diff=" << max_diff << ")\n";
}

void testReluBasic() {
    std::cout << "  Testing ReLU basic (positive, negative, zero)... " << std::flush;

    std::vector<float> data = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f, -100.0f};
    std::vector<float> cpu_data = data;
    size_t count = data.size();
    size_t bytes = count * sizeof(float);

    // CPU reference
    nevrine::cpu::reluCpuReference(cpu_data.data(), count);

    // GPU execution via arena
    nevrine::core::DeviceMemoryArena arena(bytes + 256, 256);
    float* d_data = static_cast<float*>(arena.allocate(bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));

    nevrine::cuda::launchRelu(d_data, count);
    CUDA_SYNC_AND_CHECK();

    std::vector<float> gpu_result(count);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    verifyActivation(gpu_result, cpu_data, "ReLU basic");
}

void testReluLargeTensor() {
    std::cout << "  Testing ReLU large tensor (1M elements)... " << std::flush;

    const size_t count = 1'000'000;
    const size_t bytes = count * sizeof(float);

    std::vector<float> data(count);
    // Fill with alternating positive/negative pattern
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i % 2 == 0 ? static_cast<int>(i) - 500000 : static_cast<int>(i));
    }
    std::vector<float> cpu_data = data;

    nevrine::cpu::reluCpuReference(cpu_data.data(), count);

    nevrine::core::DeviceMemoryArena arena(bytes + 256, 256);
    float* d_data = static_cast<float*>(arena.allocate(bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));

    nevrine::cuda::launchRelu(d_data, count);
    CUDA_SYNC_AND_CHECK();

    std::vector<float> gpu_result(count);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    verifyActivation(gpu_result, cpu_data, "ReLU large");
}

void testReluEmptyTensor() {
    std::cout << "  Testing ReLU empty tensor (0 elements)... " << std::flush;

    // Should not crash, just return immediately
    nevrine::cuda::launchRelu(nullptr, 0);
    std::cout << "PASSED\n";
}

void testLeakyReluBasic() {
    std::cout << "  Testing Leaky ReLU basic (negative_slope=0.1)... " << std::flush;

    std::vector<float> data = {-5.0f, -2.0f, -0.1f, 0.0f, 0.1f, 2.0f, 5.0f};
    std::vector<float> cpu_data = data;
    size_t count = data.size();
    size_t bytes = count * sizeof(float);
    float negative_slope = 0.1f;

    nevrine::cpu::leakyReluCpuReference(cpu_data.data(), count, negative_slope);

    nevrine::core::DeviceMemoryArena arena(bytes + 256, 256);
    float* d_data = static_cast<float*>(arena.allocate(bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));

    nevrine::cuda::launchLeakyRelu(d_data, count, negative_slope);
    CUDA_SYNC_AND_CHECK();

    std::vector<float> gpu_result(count);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    verifyActivation(gpu_result, cpu_data, "Leaky ReLU basic");
}

void testLeakyReluDefaultSlope() {
    std::cout << "  Testing Leaky ReLU default slope (0.01)... " << std::flush;

    std::vector<float> data = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    std::vector<float> cpu_data = data;
    size_t count = data.size();
    size_t bytes = count * sizeof(float);

    // Using default slope 0.01
    nevrine::cpu::leakyReluCpuReference(cpu_data.data(), count);

    nevrine::core::DeviceMemoryArena arena(bytes + 256, 256);
    float* d_data = static_cast<float*>(arena.allocate(bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));

    nevrine::cuda::launchLeakyRelu(d_data, count);
    CUDA_SYNC_AND_CHECK();

    std::vector<float> gpu_result(count);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    verifyActivation(gpu_result, cpu_data, "Leaky ReLU default slope");
}

void testReluInvalidPointer() {
    std::cout << "  Testing ReLU null pointer exception... " << std::flush;
    bool caught = false;
    try {
        nevrine::cuda::launchRelu(nullptr, 10);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    if (!caught) throw std::runtime_error("Failed to catch null pointer");
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "========================================================\n"
              << "       Nevrine ACTIVATIONS TEST SUITE (P4)              \n"
              << "========================================================\n";

    try {
        testReluBasic();
        testReluLargeTensor();
        testReluEmptyTensor();
        testLeakyReluBasic();
        testLeakyReluDefaultSlope();
        testReluInvalidPointer();

        std::cout << "\n[ALL TESTS PASSED] ReLU & Leaky ReLU match CPU reference across all test cases!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED] " << e.what() << "\n";
        return 1;
    }
}
