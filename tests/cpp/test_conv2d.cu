#include "nevrine/core/arena.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include "nevrine/cpu/activations_cpu.hpp"
#include "nevrine/cpu/conv2d_cpu.hpp"
#include "nevrine/cuda/conv2d.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void fillRandom(std::vector<float>& vec, float min_val = -1.0f, float max_val = 1.0f,
                unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& val : vec) {
        val = dist(rng);
    }
}

/**
 * @brief Runs a single Conv2D test case, comparing GPU output against CPU golden reference.
 *
 * @param N Batch size
 * @param C_in Input channels
 * @param H Input height
 * @param W Input width
 * @param C_out Output channels
 * @param kH Kernel height
 * @param kW Kernel width
 * @param stride Convolution stride
 * @param pad Convolution padding
 * @param use_bias Whether to include bias
 * @param fuse_relu Whether to test the fused Conv2D+ReLU variant
 * @param label Test label for output
 */
void runSingleConv2dTest(int N, int C_in, int H, int W,
                         int C_out, int kH, int kW,
                         int stride, int pad,
                         bool use_bias, bool fuse_relu,
                         const std::string& label) {
    std::cout << "  Testing " << label
              << " (N=" << N << ", C_in=" << C_in << ", H=" << H << ", W=" << W
              << ", C_out=" << C_out << ", kH=" << kH << ", kW=" << kW
              << ", stride=" << stride << ", pad=" << pad
              << ", bias=" << (use_bias ? "yes" : "no")
              << ", relu=" << (fuse_relu ? "fused" : "no") << ")... " << std::flush;

    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;

    size_t input_size = static_cast<size_t>(N) * C_in * H * W;
    size_t kernel_size = static_cast<size_t>(C_out) * C_in * kH * kW;
    size_t bias_size = use_bias ? static_cast<size_t>(C_out) : 0;
    size_t output_size = static_cast<size_t>(N) * C_out * H_out * W_out;

    size_t input_bytes = input_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    size_t bias_bytes = bias_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    // Allocate host data
    std::vector<float> h_input(input_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_bias(bias_size);
    std::vector<float> h_output_cpu(output_size, 0.0f);
    std::vector<float> h_output_gpu(output_size, 0.0f);

    fillRandom(h_input, -1.0f, 1.0f, 42);
    fillRandom(h_kernel, -0.5f, 0.5f, 123);
    if (use_bias) {
        fillRandom(h_bias, -0.1f, 0.1f, 456);
    }

    // 1. CPU golden reference
    const float* bias_ptr = use_bias ? h_bias.data() : nullptr;
    nevrine::cpu::conv2dCpuReference(h_input.data(), h_kernel.data(), bias_ptr,
                                     h_output_cpu.data(),
                                     N, C_in, H, W, C_out, kH, kW, stride, pad);

    // If testing fused ReLU, apply ReLU to CPU reference output
    if (fuse_relu) {
        nevrine::cpu::reluCpuReference(h_output_cpu.data(), output_size);
    }

    // 2. Allocate GPU memory via arena
    size_t arena_capacity = input_bytes + kernel_bytes + bias_bytes + output_bytes + 8192;
    nevrine::core::DeviceMemoryArena arena(arena_capacity, 256);

    float* d_input = static_cast<float*>(arena.allocate(input_bytes));
    float* d_kernel = static_cast<float*>(arena.allocate(kernel_bytes));
    float* d_bias = nullptr;
    if (use_bias) {
        d_bias = static_cast<float*>(arena.allocate(bias_bytes));
    }
    float* d_output = static_cast<float*>(arena.allocate(output_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));
    if (use_bias) {
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

    // 3. Run GPU kernel
    if (fuse_relu) {
        nevrine::cuda::launchConv2dRelu(d_input, d_kernel, d_bias, d_output,
                                         N, C_in, H, W, C_out, kH, kW, stride, pad);
    } else {
        nevrine::cuda::launchConv2d(d_input, d_kernel, d_bias, d_output,
                                     N, C_in, H, W, C_out, kH, kW, stride, pad);
    }
    CUDA_SYNC_AND_CHECK();

    // 4. Transfer back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    // 5. Numerical verification
    float max_diff = 0.0f;
    float tolerance = 1e-4f;

    for (size_t i = 0; i < output_size; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            std::cerr << "FAILED! Mismatch at element " << i
                      << ": GPU=" << h_output_gpu[i] << ", CPU=" << h_output_cpu[i]
                      << ", diff=" << diff << "\n";
            throw std::runtime_error(label + " numerical tolerance exceeded");
        }
    }

    std::cout << "PASSED (max_diff=" << max_diff << ")\n";
}

int main() {
    std::cout << "========================================================\n"
              << "       Nevrine CONV2D TEST SUITE (P4)                   \n"
              << "========================================================\n";

    try {
        // ─── 1×1 Convolution (Pointwise) ─────────────────────────────────
        runSingleConv2dTest(1, 3, 8, 8, 16, 1, 1, 1, 0,
                           false, false, "1x1 Pointwise Conv");

        // ─── 3×3 Convolution, No Padding ─────────────────────────────────
        runSingleConv2dTest(1, 1, 8, 8, 1, 3, 3, 1, 0,
                           false, false, "3x3 Conv, pad=0");

        // ─── 3×3 Convolution, Same Padding ───────────────────────────────
        runSingleConv2dTest(1, 1, 8, 8, 1, 3, 3, 1, 1,
                           false, false, "3x3 Conv, pad=1 (same)");

        // ─── 5×5 Kernel, Stride=2 ────────────────────────────────────────
        runSingleConv2dTest(1, 1, 16, 16, 1, 5, 5, 2, 2,
                           false, false, "5x5 Conv, stride=2, pad=2");

        // ─── Non-Square Input (Rectangular) ──────────────────────────────
        runSingleConv2dTest(1, 1, 7, 13, 1, 3, 3, 1, 0,
                           false, false, "Non-Square Input (7x13)");

        // ─── Non-Square Kernel ───────────────────────────────────────────
        runSingleConv2dTest(1, 1, 10, 10, 1, 3, 5, 1, 0,
                           false, false, "Non-Square Kernel (3x5)");

        // ─── Multi-Batch ─────────────────────────────────────────────────
        runSingleConv2dTest(4, 1, 8, 8, 1, 3, 3, 1, 1,
                           false, false, "Multi-Batch (N=4)");

        // ─── Multi-Channel Input & Output ────────────────────────────────
        runSingleConv2dTest(2, 3, 16, 16, 8, 3, 3, 1, 1,
                           false, false, "Multi-Channel (C_in=3, C_out=8)");

        // ─── Single-Pixel Output ─────────────────────────────────────────
        runSingleConv2dTest(1, 1, 3, 3, 1, 3, 3, 1, 0,
                           false, false, "Single-Pixel Output (3x3 input, 3x3 kernel)");

        // ─── Bias Addition ───────────────────────────────────────────────
        runSingleConv2dTest(1, 3, 8, 8, 4, 3, 3, 1, 1,
                           true, false, "Conv + Bias");

        // ─── Fused Conv2D + ReLU ─────────────────────────────────────────
        runSingleConv2dTest(2, 3, 16, 16, 8, 3, 3, 1, 1,
                           true, true, "Fused Conv2D+ReLU");

        // ─── Fused Conv2D + ReLU, Larger ─────────────────────────────────
        runSingleConv2dTest(4, 16, 32, 32, 32, 3, 3, 1, 1,
                           true, true, "Fused Conv2D+ReLU (larger)");

        // ─── Prime Dimensions Stress Test ────────────────────────────────
        runSingleConv2dTest(1, 1, 11, 13, 1, 3, 3, 1, 0,
                           false, false, "Prime Dimensions (11x13)");

        // ─── Stride > Kernel Edge Case ───────────────────────────────────
        runSingleConv2dTest(1, 1, 16, 16, 1, 3, 3, 4, 0,
                           false, false, "Stride > Kernel (stride=4, k=3)");

        // ─── Invalid Arguments Exception Handling ────────────────────────
        std::cout << "  Testing Invalid Dimensions Exception... " << std::flush;
        bool caught = false;
        try {
            nevrine::cuda::launchConv2d(nullptr, nullptr, nullptr, nullptr,
                                         0, 1, 8, 8, 1, 3, 3, 1, 0);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) throw std::runtime_error("Failed to catch invalid dimensions");
        std::cout << "PASSED\n";

        std::cout << "\n[ALL TESTS PASSED] Conv2D & Fused Conv2D+ReLU match CPU reference across all test cases!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED] " << e.what() << "\n";
        return 1;
    }
}
