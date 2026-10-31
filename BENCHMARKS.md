# Nevrine Benchmark Report & Empirical Evaluation

**Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM, 24 SMs, Ada Lovelace `sm_89`)  
**Driver / CUDA**: NVIDIA Driver 591.66 | CUDA Toolkit 12.4 (`nvcc V12.4.131`)  
**Host Compiler / OS**: GNU g++ 15.2.0 | Linux 6.6.87.2-WSL2 (Ubuntu)  
**Methodology**: High-resolution CUDA events (`cudaEventElapsedTime`), 5 warm-up iterations, 20 measurement iterations. Median latency and 95th-percentile (p95) reported.

---

## 1. Matrix Multiplication (SGEMM) Benchmarks

### 1.1 Naive SGEMM Baseline (P2 Milestone)
- **Kernel Implementation**: Direct global memory reads per thread, 2D block launch configuration ($16 \times 16$), no shared memory caching or register tiling.
- **Arithmetic Intensity**: Low (every multiply-accumulate loads from global memory).

| Matrix Dimensions ($M \times N \times K$) | Total FLOPs ($2 \cdot M \cdot N \cdot K$) | Median Latency (ms) | p95 Latency (ms) | Throughput (GFLOP/s) | Correctness Pass |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$256 \times 256 \times 256$** | 33.55 MFLOPs | 0.1147 ms | 0.2580 ms | 292.57 GFLOP/s | ✅ Passed ($< 10^{-6}$) |
| **$512 \times 512 \times 512$** | 268.44 MFLOPs | 0.3983 ms | 0.4421 ms | 673.89 GFLOP/s | ✅ Passed ($< 10^{-6}$) |
| **$1024 \times 1024 \times 1024$** | 2.147 GFLOPs | 3.0667 ms | 3.7345 ms | 700.27 GFLOP/s | ✅ Passed ($< 10^{-5}$) |
| **$2048 \times 2048 \times 2048$** | 17.180 GFLOPs | 19.4785 ms | 24.7715 ms | 881.99 GFLOP/s | ✅ Passed ($< 10^{-5}$) |

### 1.2 Shared-Memory Tiled SGEMM vs. cuBLAS (P3 Milestone)
- **Kernel Implementation**: 2D shared-memory tiling ($32 \times 32$) with $+1$ column padding (`s_A[32][33]`, `s_B[32][33]`) to mitigate 32-bank conflicts, `#pragma unroll` on inner reductions, and full boundary predicates for arbitrary matrix dimensions.
- **cuBLAS Reference**: Highly optimized vendor GEMM invoked via row-major algebraic mapping ($C = A \cdot B \iff C^T = B^T \cdot A^T$) on `cublasSgemm`.
- **Timing**: CUDA event elapsed time across 20 measured iterations after 5 warm-up iterations.

| Dimensions ($M \times N \times K$) | Naive Latency | Tiled Latency | cuBLAS Latency | Naive GFLOP/s | Tiled GFLOP/s | cuBLAS GFLOP/s | % of cuBLAS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$256 \times 256 \times 256$** | 0.0594 ms | 0.0655 ms | 0.0143 ms | 564.97 GFLOP/s | 512.00 GFLOP/s | 2345.81 GFLOP/s | **21.83%** |
| **$512 \times 512 \times 512$** | 0.3871 ms | 0.4577 ms | 0.0532 ms | 693.50 GFLOP/s | 586.45 GFLOP/s | 5041.23 GFLOP/s | **11.63%** |
| **$1024 \times 1024 \times 1024$** | 3.2532 ms | 4.1114 ms | 0.3491 ms | 660.10 GFLOP/s | 522.33 GFLOP/s | 6151.13 GFLOP/s | **8.49%** |
| **$2048 \times 2048 \times 2048$** | 24.3169 ms | 23.6514 ms | 1.9025 ms | 706.50 GFLOP/s | 726.38 GFLOP/s | 9030.17 GFLOP/s | **8.04%** |

> [!NOTE]
> Standard shared-memory tiling reduces DRAM traffic by $32\times$. Achieving $>50\%$ of cuBLAS's 9.0 TFLOP/s ceiling requires register tiling (2D blocktiling computing 4x4 or 8x8 per thread to amortize shared memory reads) and vector loads (`float4`), targeted for advanced kernel optimization in Phase 7.

---

## 2. Memory Arena Evaluation (P1 Milestone)

- **Policy**: Monolithic 256-byte aligned bump allocator on GPU VRAM.
- **Steady-State Invariant**: Zero `cudaMalloc` or `cudaFree` calls during execution.
- **Verification**:
  - Alignment verified across irregular allocation sizes (1, 3, 17, 64, 127, 255, 256, 513, 1024 bytes).
  - Out-of-memory capacity boundaries enforced with `ArenaOutOfMemoryException`.
  - Multi-region GPU kernel execution verified: 3 distinct tensors sub-allocated within the arena, processed asynchronously by GPU kernels, verified against host without neighbor corruption.

---

## 3. End-to-End CNN Pipeline & Serving Latency (P6 Milestone)

Evaluation of `ConvNetClassifier` (Conv2D + Fused ReLU $\to$ MaxPool2D $\to$ Conv2D + Fused ReLU $\to$ MaxPool2D $\to$ GEMM + ReLU $\to$ GEMM Linear Layer $\to$ Softmax) measured on Ada Lovelace RTX 4060:

| Batch Size | Execution Backend | Median Latency (ms) | p95 Latency (ms) | Throughput (samples/s) | GPU Speedup vs CPU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | `cuda_tiled` | 1.91 ms | 2.70 ms | 523.7 samples/s | ~1.0x (Kernel launch overhead dominated) |
| **Batch 1** | `cublas` | 2.28 ms | 2.80 ms | 438.1 samples/s | 0.81x |
| **Batch 1** | `cpu` | 1.85 ms | 2.39 ms | 539.5 samples/s | Baseline |
| **Batch 4** | `cuda_tiled` | 2.98 ms | 3.53 ms | 1340.5 samples/s | **1.95x** |
| **Batch 4** | `cublas` | 2.90 ms | 3.55 ms | 1379.3 samples/s | **2.00x** |
| **Batch 4** | `cpu` | 5.80 ms | 6.51 ms | 689.7 samples/s | Baseline |
| **Batch 16** | `cuda_tiled` | 6.09 ms | 6.85 ms | 2625.8 samples/s | **3.59x** |
| **Batch 16** | `cublas` | 6.14 ms | 7.10 ms | 2603.9 samples/s | **3.56x** |
| **Batch 16** | `cpu` | 21.86 ms | 25.52 ms | 731.8 samples/s | Baseline |

### Key Serving Invariants
- **Pre-warming Lifecycle**: Zero cold-start latency spike on first user request (warmup pass executes in FastAPI lifespan handler).
- **In-flight Telemetry**: Real-time VRAM allocation, compute utilization, and percentile distributions (`/health`, `/device`, `/benchmark`).

