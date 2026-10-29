#include "nevrine/cuda/conv2d.hpp"
#include <stdexcept>

namespace nevrine {
namespace cuda {

namespace {

constexpr int CONV_BLOCK_X = 16;
constexpr int CONV_BLOCK_Y = 16;

/**
 * @brief Templated naive Conv2D forward kernel.
 *
 * Template parameter `ApplyReLU` enables kernel fusion: when true, the output
 * is clamped to max(0, x) in the same kernel pass, eliminating a separate
 * activation launch and one full-tensor global memory read/write cycle.
 *
 * Thread mapping:
 *   threadIdx.x -> output width  (ow)
 *   threadIdx.y -> output height (oh)
 *   blockIdx.z  -> (batch_index * C_out + channel_out)
 */
template <bool ApplyReLU>
__global__ void conv2dKernel(const float* __restrict__ input,
                             const float* __restrict__ kernel_weights,
                             const float* __restrict__ bias,
                             float* __restrict__ output,
                             int N, int C_in, int H, int W,
                             int C_out, int kH, int kW,
                             int H_out, int W_out,
                             int stride, int pad) {
    int ow = blockDim.x * blockIdx.x + threadIdx.x;
    int oh = blockDim.y * blockIdx.y + threadIdx.y;

    // Decode batch index and output channel from grid.z
    int n_co = blockIdx.z;
    int n  = n_co / C_out;
    int co = n_co % C_out;

    // Boundary check: exit early if this thread is outside the output volume
    if (ow >= W_out || oh >= H_out || n >= N) return;

    float acc = 0.0f;

    // Accumulate across all input channels and kernel spatial positions
    for (int ci = 0; ci < C_in; ++ci) {
        for (int fh = 0; fh < kH; ++fh) {
            for (int fw = 0; fw < kW; ++fw) {
                int ih = oh * stride + fh - pad;
                int iw = ow * stride + fw - pad;

                // Implicit zero-padding: only accumulate if within input bounds
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int input_idx = ((n * C_in + ci) * H + ih) * W + iw;
                    int kernel_idx = ((co * C_in + ci) * kH + fh) * kW + fw;
                    acc += input[input_idx] * kernel_weights[kernel_idx];
                }
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        acc += bias[co];
    }

    // Fused ReLU: apply activation in-register before writing to global memory
    if constexpr (ApplyReLU) {
        acc = acc > 0.0f ? acc : 0.0f;
    }

    int output_idx = ((n * C_out + co) * H_out + oh) * W_out + ow;
    output[output_idx] = acc;
}

// Validates Conv2D arguments and computes output dimensions
void validateConv2dArgs(const float* input, const float* kernel_weights, const float* output,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride, int pad,
                        int& H_out, int& W_out) {
    if (N <= 0 || C_in <= 0 || H <= 0 || W <= 0 || C_out <= 0 || kH <= 0 || kW <= 0) {
        throw std::invalid_argument("Conv2D dimensions must be strictly positive");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Conv2D stride must be strictly positive");
    }
    if (pad < 0) {
        throw std::invalid_argument("Conv2D padding must be non-negative");
    }
    if (input == nullptr || kernel_weights == nullptr || output == nullptr) {
        throw std::invalid_argument("Conv2D input, kernel, and output pointers must not be null");
    }

    H_out = (H + 2 * pad - kH) / stride + 1;
    W_out = (W + 2 * pad - kW) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("Conv2D output dimensions are non-positive; check input size, kernel size, stride, and padding");
    }
}

} // anonymous namespace

void launchConv2d(const float* input, const float* kernel_weights,
                  const float* bias, float* output,
                  int N, int C_in, int H, int W,
                  int C_out, int kH, int kW,
                  int stride, int pad,
                  cudaStream_t stream) {
    int H_out, W_out;
    validateConv2dArgs(input, kernel_weights, output, N, C_in, H, W, C_out, kH, kW, stride, pad, H_out, W_out);

    dim3 block(CONV_BLOCK_X, CONV_BLOCK_Y);
    dim3 grid((W_out + CONV_BLOCK_X - 1) / CONV_BLOCK_X,
              (H_out + CONV_BLOCK_Y - 1) / CONV_BLOCK_Y,
              N * C_out);

    conv2dKernel<false><<<grid, block, 0, stream>>>(
        input, kernel_weights, bias, output,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, pad);
    CUDA_CHECK_KERNEL("conv2dKernel<false>");
}

void launchConv2dRelu(const float* input, const float* kernel_weights,
                      const float* bias, float* output,
                      int N, int C_in, int H, int W,
                      int C_out, int kH, int kW,
                      int stride, int pad,
                      cudaStream_t stream) {
    int H_out, W_out;
    validateConv2dArgs(input, kernel_weights, output, N, C_in, H, W, C_out, kH, kW, stride, pad, H_out, W_out);

    dim3 block(CONV_BLOCK_X, CONV_BLOCK_Y);
    dim3 grid((W_out + CONV_BLOCK_X - 1) / CONV_BLOCK_X,
              (H_out + CONV_BLOCK_Y - 1) / CONV_BLOCK_Y,
              N * C_out);

    conv2dKernel<true><<<grid, block, 0, stream>>>(
        input, kernel_weights, bias, output,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, pad);
    CUDA_CHECK_KERNEL("conv2dKernel<true>");
}

} // namespace cuda
} // namespace nevrine
