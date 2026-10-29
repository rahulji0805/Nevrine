#pragma once

#include "nevrine/core/cuda_utils.hpp"
#include <cuda_runtime.h>

namespace nevrine {
namespace cuda {

/**
 * @brief Launches naive 2D convolution forward kernel on the GPU.
 *
 * Layout: NCHW (batch × channels × height × width)
 * Computes: output[n][co][oh][ow] = bias[co] + sum_ci sum_kh sum_kw { input[...] * kernel[...] }
 *
 * Thread mapping: one thread per output pixel (n, co, oh, ow).
 * Launch: 2D blocks (16×16) over spatial (W_out, H_out), grid.z covers batch × output channels.
 * Implicit zero-padding with full boundary predicates.
 *
 * @param input Device pointer to input tensor (N, C_in, H, W)
 * @param kernel Device pointer to kernel tensor (C_out, C_in, kH, kW)
 * @param bias Device pointer to bias vector (C_out), or nullptr if no bias
 * @param output Device pointer to output tensor (N, C_out, H_out, W_out)
 * @param N Batch size
 * @param C_in Number of input channels
 * @param H Input height
 * @param W Input width
 * @param C_out Number of output channels (filters)
 * @param kH Kernel height
 * @param kW Kernel width
 * @param stride Convolution stride (default 1)
 * @param pad Zero-padding applied symmetrically (default 0)
 * @param stream CUDA stream for asynchronous launch (default 0)
 */
void launchConv2d(const float* input, const float* kernel,
                  const float* bias, float* output,
                  int N, int C_in, int H, int W,
                  int C_out, int kH, int kW,
                  int stride = 1, int pad = 0,
                  cudaStream_t stream = 0);

/**
 * @brief Launches fused Conv2D + ReLU kernel on the GPU.
 *
 * Identical to launchConv2d, but applies ReLU(max(0, x)) to each output element
 * immediately after accumulation, avoiding a separate activation pass and
 * eliminating one full-tensor global memory round-trip.
 *
 * @param input Device pointer to input tensor (N, C_in, H, W)
 * @param kernel Device pointer to kernel tensor (C_out, C_in, kH, kW)
 * @param bias Device pointer to bias vector (C_out), or nullptr if no bias
 * @param output Device pointer to output tensor (N, C_out, H_out, W_out)
 * @param N Batch size
 * @param C_in Number of input channels
 * @param H Input height
 * @param W Input width
 * @param C_out Number of output channels (filters)
 * @param kH Kernel height
 * @param kW Kernel width
 * @param stride Convolution stride (default 1)
 * @param pad Zero-padding applied symmetrically (default 0)
 * @param stream CUDA stream for asynchronous launch (default 0)
 */
void launchConv2dRelu(const float* input, const float* kernel,
                      const float* bias, float* output,
                      int N, int C_in, int H, int W,
                      int C_out, int kH, int kW,
                      int stride = 1, int pad = 0,
                      cudaStream_t stream = 0);

} // namespace cuda
} // namespace nevrine
