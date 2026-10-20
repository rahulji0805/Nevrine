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

---

## 2. Memory Arena Evaluation (P1 Milestone)

- **Policy**: Monolithic 256-byte aligned bump allocator on GPU VRAM.
- **Steady-State Invariant**: Zero `cudaMalloc` or `cudaFree` calls during execution.
- **Verification**:
  - Alignment verified across irregular allocation sizes (1, 3, 17, 64, 127, 255, 256, 513, 1024 bytes).
  - Out-of-memory capacity boundaries enforced with `ArenaOutOfMemoryException`.
  - Multi-region GPU kernel execution verified: 3 distinct tensors sub-allocated within the arena, processed asynchronously by GPU kernels, verified against host without neighbor corruption.
