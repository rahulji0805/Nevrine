#include "nevrine/cpu/conv2d_cpu.hpp"
#include <stdexcept>

namespace nevrine {
namespace cpu {

void conv2dCpuReference(const float* input, const float* kernel,
                        const float* bias, float* output,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride, int pad) {
    if (N <= 0 || C_in <= 0 || H <= 0 || W <= 0 || C_out <= 0 || kH <= 0 || kW <= 0) {
        throw std::invalid_argument("Conv2D dimensions must be strictly positive");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Conv2D stride must be strictly positive");
    }
    if (pad < 0) {
        throw std::invalid_argument("Conv2D padding must be non-negative");
    }
    if (input == nullptr || kernel == nullptr || output == nullptr) {
        throw std::invalid_argument("Conv2D input, kernel, and output pointers must not be null");
    }

    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("Conv2D output dimensions are non-positive; check input size, kernel size, stride, and padding");
    }

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float acc = 0.0f;

                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int fh = 0; fh < kH; ++fh) {
                            for (int fw = 0; fw < kW; ++fw) {
                                int ih = oh * stride + fh - pad;
                                int iw = ow * stride + fw - pad;

                                // Implicit zero-padding: skip out-of-bounds
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int input_idx = ((n * C_in + ci) * H + ih) * W + iw;
                                    int kernel_idx = ((co * C_in + ci) * kH + fh) * kW + fw;
                                    acc += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }

                    // Add bias if provided
                    if (bias != nullptr) {
                        acc += bias[co];
                    }

                    int output_idx = ((n * C_out + co) * H_out + oh) * W_out + ow;
                    output[output_idx] = acc;
                }
            }
        }
    }
}

} // namespace cpu
} // namespace nevrine
