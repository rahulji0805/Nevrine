# NEXUS-CUDA: Cloud-Native Neural Acceleration Engine

A lightweight, transparent C++/CUDA neural network inference runtime engineered for predictable memory behavior, high-throughput kernel execution, and zero steady-state GPU allocations.

---

## 🎯 Project Roadmap & Status

Based on the 8-week execution plan:

| Milestone | Phase | Description | Status |
| :--- | :--- | :--- | :--- |
| **P0** | **Session 1** | Build foundation, CMake toolchain, device info & CUDA smoke test | **Completed** ✅ |
| **P1** | **Session 2** | VRAM Arena skeleton, 256-byte alignment, offset sub-allocation, bounds & exhaustion tests | **Next Up** ⏳ |
| **P2** | **Session 3** | Naive GEMM baseline kernel, launch bounds, CPU reference verification | Planned |
| **P3** | **Week 3** | Shared-memory Tiled GEMM, CUDA-event timing, benchmarks vs cuBLAS | Planned |
| **P4** | **Week 4** | Conv2D kernel, activations (ReLU), edge-case validation | Planned |
| **P5** | **Week 5** | Pybind11 Python bindings, explicit tensor transfers, CPU fallback | Planned |
| **P6** | **Week 6-7** | End-to-end CNN pipeline, FastAPI serving, Docker container, AWS deployment | Planned |
| **P7** | **Week 8** | Nsight Systems & Nsight Compute profiling, BENCHMARKS.md publication | Planned |

---

## 🏗️ Repository Architecture

```text
Nevrine/
├── CMakeLists.txt            # Build configuration (C++17, CUDA 12.4, sm_89 + sm_75)
├── include/nexus/
│   ├── core/
│   │   ├── cuda_utils.hpp    # CUDA_CHECK, CUDA_CHECK_KERNEL, CudaException
│   │   └── device_info.hpp   # Device properties & VRAM memory query API
│   └── cuda/
│       └── smoke_kernel.hpp  # CUDA smoke test interface
├── src/
│   ├── core/
│   │   └── device_info.cpp   # DeviceManager implementation
│   ├── cuda/
│   │   └── smoke_kernel.cu   # CUDA elementwise affine kernel
│   └── main.cpp              # Environment verification entry point
├── docs/
│   ├── DESIGN.md             # Memory arena contract, ownership rules & invariants
│   └── environment.md        # Hardware and software toolchain snapshot
└── README.md
```

---

## 🚀 Quick Start

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
./build/nexus_smoke_test
```

---

## 📜 Documentation Links
- [DESIGN.md](file:///docs/DESIGN.md): Memory arena contract, boundary of guarantees, and alignment rules.
- [environment.md](file:///docs/environment.md): Detailed GPU hardware specs, driver versions, and test results.
