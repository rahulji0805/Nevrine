# Nevrine: Cloud-Native Neural Acceleration Engine

A lightweight, transparent C++/CUDA neural network inference runtime engineered for predictable memory behavior, high-throughput kernel execution, and zero steady-state GPU allocations.

---

##  Project Roadmap & Status

Based on the 8-week execution plan:

| Milestone | Phase | Description | Status |
| :--- | :--- | :--- | :--- |
| **P0** | **Session 1** | Build foundation, CMake toolchain, device info & CUDA smoke test | **Completed**  |
| **P1** | **Session 2** | VRAM Arena skeleton, 256-byte alignment, offset sub-allocation, bounds & exhaustion tests | **Completed**  |
| **P2** | **Session 3** | Naive GEMM baseline kernel, launch bounds, CPU reference verification & benchmark | **Completed**  |
| **P3** | **Week 3** | Shared-memory Tiled GEMM ($32 \times 32$), CUDA-event timing, benchmarks vs cuBLAS | **Completed**  |
| **P4** | **Week 4** | Conv2D kernel, activations (ReLU), edge-case validation | **Completed**  |
| **P5** | **Week 5** | Pybind11 Python bindings, explicit tensor transfers, CPU fallback | **Completed**  |
| **P6** | **Week 6-7** | End-to-end CNN pipeline, FastAPI serving, Docker container, AWS deployment | **Completed**  |
| **P7** | **Week 8** | Nsight Systems & Nsight Compute profiling, BENCHMARKS.md publication | **Next Up**  |

---

##  Performance Benchmarks (P3 Milestone)

Empirical single-precision GEMM throughput measured on **NVIDIA RTX 4060 Laptop (sm_89, Ada Lovelace)**:

| Dimensions ($M \times N \times K$) | Naive Baseline | Tiled ($32 \times 32$ Shared Memory) | cuBLAS (Vendor Peak) | % of cuBLAS |
| :--- | :--- | :--- | :--- | :--- |
| **$256 \times 256 \times 256$** | 564.97 GFLOP/s (0.059 ms) | 512.00 GFLOP/s (0.065 ms) | 2345.81 GFLOP/s (0.014 ms) | **21.83%** |
| **$512 \times 512 \times 512$** | 693.50 GFLOP/s (0.387 ms) | 586.45 GFLOP/s (0.457 ms) | 5041.23 GFLOP/s (0.053 ms) | **11.63%** |
| **$1024 \times 1024 \times 1024$** | 660.10 GFLOP/s (3.253 ms) | 522.33 GFLOP/s (4.111 ms) | 6151.13 GFLOP/s (0.349 ms) | **8.49%** |
| **$2048 \times 2048 \times 2048$** | 706.50 GFLOP/s (24.316 ms) | 726.38 GFLOP/s (23.651 ms) | 9030.17 GFLOP/s (1.902 ms) | **8.04%** |

*Detailed timing distributions, p95 latencies, and memory invariants are documented in [`BENCHMARKS.md`](file:///BENCHMARKS.md).*

---

##  Repository Architecture

```text
Nevrine/
├── CMakeLists.txt            # Build configuration (C++17, CUDA 12.4, sm_89 + sm_75, Pybind11, cuBLAS)
├── include/nevrine/
│   ├── core/
│   │   ├── arena.hpp         # Monolithic 256-byte aligned DeviceMemoryArena
│   │   ├── cuda_utils.hpp    # CUDA_CHECK, CUBLAS_CHECK, CUDA_SYNC_AND_CHECK
│   │   └── device_info.hpp   # Device properties & VRAM memory query API
│   ├── cpu/
│   │   ├── gemm_cpu.hpp      # CPU golden reference for GEMM verification
│   │   ├── activations_cpu.hpp  # CPU golden reference for ReLU / Leaky ReLU
│   │   └── conv2d_cpu.hpp    # CPU golden reference for Conv2D forward pass
│   └── cuda/
│       ├── smoke_kernel.hpp  # CUDA smoke test interface
│       ├── gemm_naive.hpp    # Naive SGEMM baseline
│       ├── gemm_tiled.hpp    # 2D shared-memory tiled SGEMM
│       ├── cublas_gemm.hpp   # cuBLAS row-major SGEMM wrapper
│       ├── activations.hpp   # ReLU & Leaky ReLU CUDA kernels
│       └── conv2d.hpp        # Naive Conv2D & fused Conv2D+ReLU kernels
├── src/
│   ├── core/
│   │   ├── arena.cpp         # Memory arena implementation
│   │   └── device_info.cpp   # DeviceManager implementation
│   ├── cpu/
│   │   ├── gemm_cpu.cpp      # Single-threaded CPU GEMM reference
│   │   ├── activations_cpu.cpp  # CPU ReLU / Leaky ReLU reference
│   │   └── conv2d_cpu.cpp    # CPU Conv2D forward pass reference
│   ├── cuda/
│   │   ├── smoke_kernel.cu   # CUDA elementwise affine kernel
│   │   ├── gemm_naive.cu     # Naive SGEMM kernel
│   │   ├── gemm_tiled.cu     # Tiled SGEMM with shared memory + bank padding
│   │   ├── cublas_gemm.cu    # cuBLAS handle management & dispatch
│   │   ├── activations.cu    # ReLU & Leaky ReLU CUDA implementations
│   │   └── conv2d.cu         # Naive Conv2D + fused Conv2D+ReLU (templated)
│   ├── python/
│   │   └── bindings.cpp      # Pybind11 Python bindings
│   └── main.cpp              # Environment verification entry point
├── python/nevrine/
│   ├── __init__.py           # Idiomatic Python package wrapper
│   ├── _nevrine.so           # Compiled C++/CUDA extension module
│   └── models/
│       ├── __init__.py       # Model architectures export
│       └── cnn.py            # ConvNetClassifier pipeline (Conv2D+ReLU, MaxPool, GEMM)
├── serving/
│   ├── app.py                # FastAPI inference service with lifespan pre-warming
│   └── schemas.py            # Pydantic request & response models
├── docker/
│   ├── Dockerfile            # Multi-stage CUDA 12.4 + Python 3.10 runtime container
│   └── docker-compose.yml    # GPU-accelerated Docker Compose service
├── deploy/
│   └── aws/
│       ├── DEPLOYMENT.md     # Production deployment runbook for EC2 g4dn (Tesla T4)
│       ├── cloud-init.yaml   # Automated cloud-init provisioning & setup
│       └── nevrine.service   # Systemd unit file for production serving
├── benchmarks/
│   ├── bench_gemm_naive.cu   # P2 baseline benchmark harness
│   └── bench_gemm_p3.cu      # P3 3-way benchmark (Naive vs Tiled vs cuBLAS)
├── tests/
│   ├── cpp/
│   │   ├── test_arena.cu         # Memory arena alignment & boundary tests
│   │   ├── test_gemm.cu          # Naive GEMM correctness tests
│   │   ├── test_gemm_tiled.cu    # Tiled GEMM & cuBLAS correctness test suite
│   │   ├── test_activations.cu   # ReLU & Leaky ReLU correctness tests
│   │   └── test_conv2d.cu        # Conv2D & fused Conv2D+ReLU test suite (15 cases)
│   └── python/
│       ├── test_device.py        # Python device detection and memory metrics
│       ├── test_arena.py         # DeviceMemoryArena Python tests
│       ├── test_gemm.py          # GEMM tests across 4 backends
│       ├── test_conv2d.py        # Conv2D + fused ReLU Python tests
│       ├── test_activations.py   # ReLU & Leaky ReLU Python tests
│       ├── test_pipeline.py      # End-to-end CNN pipeline & pooling tests
│       └── test_serving.py       # FastAPI endpoint tests (/health, /predict, /benchmark)
├── docs/
│   ├── DESIGN.md             # Memory arena contract, ownership rules & invariants
│   └── environment.md        # Hardware and software toolchain snapshot
├── BENCHMARKS.md             # Empirical latency & GFLOP/s evaluation
└── README.md
```

---

##  Quick Start

### Python Usage
```python
import nevrine
import numpy as np

# 1. Inspect Hardware
print(nevrine.get_device_properties())

# 2. Fast Tiled GEMM on GPU
A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)
C = nevrine.gemm(A, B, backend="cuda_tiled")

# 3. Fused Conv2D + ReLU
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
w = np.random.randn(16, 3, 3, 3).astype(np.float32)
out = nevrine.conv2d(x, w, stride=1, pad=1, fuse_relu=True)

# 4. End-to-End Deep CNN Classifier
from nevrine.models import ConvNetClassifier
classifier = ConvNetClassifier(backend="cuda_tiled")
img_batch = np.random.randn(4, 1, 28, 28).astype(np.float32)
predictions = classifier.predict(img_batch)
print("Batch predictions:", predictions)
```

### FastAPI Inference Serving
```bash
# Start production-ready serving daemon
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Telemetry & VRAM utilization check
curl -s http://localhost:8000/health | jq .

# In-flight percentile benchmark
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"iterations": 100, "warmup": 10, "batch_size": 1, "backend": "cuda_tiled"}' | jq .
```

### Cloud & Docker Deployment
```bash
# Launch container with NVIDIA Container Toolkit
docker compose -f docker/docker-compose.yml up -d --build
```
For AWS EC2 (`g4dn.xlarge` NVIDIA T4) automated provisioning via `cloud-init` and `systemd`, see [`deploy/aws/DEPLOYMENT.md`](file:///deploy/aws/DEPLOYMENT.md).

### Prerequisites
- NVIDIA GPU (tested on RTX 4060 8 GB, compatible with AWS T4)
- CUDA Toolkit 12.x / 13.x
- C++17 compliant compiler (`g++` or `clang++` or `MSVC`)
- CMake >= 3.20 & Ninja

### Build
```bash
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run Environment Smoke Test
```bash
./build/nevrine_smoke
```

---

## Documentation Links
- [DESIGN.md](file:///docs/DESIGN.md): Memory arena contract, boundary of guarantees, and alignment rules.
- [environment.md](file:///docs/environment.md): Detailed GPU hardware specs, driver versions, and test results.
- [DEPLOYMENT.md](file:///deploy/aws/DEPLOYMENT.md): AWS EC2 `g4dn.xlarge` (Tesla T4) cloud deployment & runbook.
