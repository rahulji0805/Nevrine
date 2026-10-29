#pragma once

#include <cstddef>

namespace nevrine {
namespace cpu {

/**
 * @brief CPU golden reference: 2D convolution forward pass.
 *
 * Layout: NCHW (batch × channels × height × width)
 * Computes: output[n][co][oh][ow] = bias[co] + sum_ci sum_kh sum_kw { input[n][ci][oh*stride+kh-pad][ow*stride+kw-pad] * kernel[co][ci][kh][kw] }
 *
 * Implicit zero-padding: out-of-bounds input reads are treated as zero.
 *
 * @param input Host pointer to input tensor (N, C_in, H, W)
 * @param kernel Host pointer to kernel tensor (C_out, C_in, kH, kW)
 * @param bias Host pointer to bias vector (C_out), or nullptr if no bias
 * @param output Host pointer to output tensor (N, C_out, H_out, W_out)
 * @param N Batch size
 * @param C_in Number of input channels
 * @param H Input height
 * @param W Input width
 * @param C_out Number of output channels (number of filters)
 * @param kH Kernel height
 * @param kW Kernel width
 * @param stride Convolution stride (default 1)
 * @param pad Zero-padding applied symmetrically (default 0)
 */
void conv2dCpuReference(const float* input, const float* kernel,
                        const float* bias, float* output,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride = 1, int pad = 0);

} // namespace cpu
} // namespace nevrine
