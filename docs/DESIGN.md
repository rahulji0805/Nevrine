# NEXUS-CUDA Architecture & Memory Design Note

## 1. Scope and Mission
NEXUS-CUDA is a lightweight, low-overhead neural acceleration inference runtime. Rather than providing an expansive autograd engine or general-purpose compiler, NEXUS-CUDA focuses on deterministic, predictable memory behavior and zero steady-state GPU allocations during inference.

---

## 2. Memory Arena Contract (P1 Specification)

### 2.1 The Problem
In standard deep learning runtimes (such as naive PyTorch or ONNX Runtime configurations), executing an inference forward pass often involves dynamic driver calls (`cudaMalloc` and `cudaFree`) or sub-optimal caching allocators that incur:
1. Non-deterministic driver call latency (tens of microseconds per invocation).
2. Host-device synchronization barriers.
3. Virtual memory fragmentation over millions of continuous API requests.

### 2.2 The NEXUS-CUDA Guarantee
NEXUS-CUDA enforces a **Fixed-Capacity Device Memory Arena**:
- **Initialization**: Upon engine initialization, a single contiguous chunk of device VRAM (e.g., 2 GB or 4 GB, user-configurable) is allocated via `cudaMalloc`.
- **Steady-State Invariance**: During the inference loop, **zero** `cudaMalloc` or `cudaFree` calls are executed.
- **Sub-allocation**: All tensor allocations (input buffers, weights, intermediate activation tensors, workspace scratchpads) are sub-allocated from this monolithic buffer.
- **Alignment**: Every sub-allocation is guaranteed to be strictly aligned to **256 bytes**. This matches NVIDIA warp memory transaction boundaries (enabling full 128-byte/256-byte coalesced loads/stores without split transactions).
- **Bounds Checking**: Every allocation request validates `offset + aligned_bytes <= capacity`. Exceeding capacity throws an explicit `OutOfMemoryException` without corrupting memory or destabilizing the CUDA driver context.

### 2.3 Allocation Handles and Lifetimes
- **Handles**: Allocations are tracked using integer offsets or opaque handles `ArenaHandle { size_t offset, size_t size, void* device_ptr }`.
- **Lifetimes**:
  1. *Persistent (Weights & Biases)*: Allocated once during model initialization; pinned for the lifetime of the engine.
  2. *Scratchpad / Activation (Per-Request)*: Allocated during the forward pass and reset after the request completes.
- **Reset Policy**: The scratchpad region uses an offset-based bump allocator that can be reset to the watermark (`reset()`) in $O(1)$ time, eliminating per-tensor free overhead.

### 2.4 Boundary of Guarantees
What is **inside** the engine's guarantee:
- All intermediate tensors created by supported operator kernels (GEMM, Conv2D, ReLU, BiasAdd).
- Weight tensors loaded during engine boot.
- Temporary workspace buffers passed to custom kernels.

What is **outside** the engine's guarantee:
- Initial host-to-device input buffers submitted by external Python / NumPy callers (until copied into the arena).
- Internal driver allocations performed by third-party vendor libraries (e.g., cuBLAS internal descriptor scratchpads if cuBLAS is called).
- CUDA context creation and driver module loading memory.

---

## 3. Kernel Execution Model

- **Error Policy**: All CUDA API calls are wrapped in `CUDA_CHECK()`. All kernel launches are checked with `CUDA_CHECK_KERNEL()` to catch launch bounds and illegal configuration immediately.
- **Synchronization Policy**: Inference operations within a stream are executed asynchronously. Synchronization (`cudaStreamSynchronize` or `cudaDeviceSynchronize`) is only performed when transferring final outputs back to host memory or when benchmarking explicitly with CUDA events.
- **Backend Selection**: Explicit backend dispatch (`Backend::CUDA` vs `Backend::CPU`). No silent, unmonitored fallbacks in hot loops.

---

## 4. Hardware Targets

| Target | Architecture | Compute Capability | VRAM | Intended Role |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA GeForce RTX 4060** | Ada Lovelace | `sm_89` | 8 GB GDDR6 | Primary local development & kernel optimization |
| **AWS EC2 g4dn (NVIDIA T4)** | Turing | `sm_75` | 16 GB GDDR6 | Cloud deployment, packaging & remote API benchmarking |
