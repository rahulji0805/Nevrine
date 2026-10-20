#include "nexus/core/arena.hpp"
#include "nexus/core/cuda_utils.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void arenaVectorAddKernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void testAlignment() {
    std::cout << "[TEST] 1. Alignment Verification (256-byte boundaries)... ";
    constexpr size_t arena_size = 1024 * 1024; // 1 MB
    nexus::core::DeviceMemoryArena arena(arena_size, 256);

    for (size_t s : {1, 3, 17, 64, 127, 255, 256, 513, 1024}) {
        void* ptr = arena.allocate(s);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if ((addr % 256) != 0) {
            throw std::runtime_error("Sub-allocation must be 256-byte aligned");
        }
        if (!arena.contains(ptr, s)) {
            throw std::runtime_error("Arena must contain allocated pointer");
        }
    }
    std::cout << "PASSED\n";
}

void testExhaustion() {
    std::cout << "[TEST] 2. Capacity Exhaustion & Out-of-Memory Handling... ";
    constexpr size_t arena_size = 1024 * 1024; // 1 MB
    nexus::core::DeviceMemoryArena arena(arena_size, 256);

    // Allocate 700 KB
    arena.allocate(700 * 1024);

    // Try to allocate 500 KB (should exceed 1 MB)
    bool caught_expected = false;
    try {
        arena.allocate(500 * 1024);
    } catch (const nexus::core::ArenaOutOfMemoryException&) {
        caught_expected = true;
    }
    if (!caught_expected) {
        throw std::runtime_error("Allocating beyond capacity must throw ArenaOutOfMemoryException");
    }
    std::cout << "PASSED\n";
}

void testResetAndReuse() {
    std::cout << "[TEST] 3. Reset and Memory Reuse... ";
    constexpr size_t arena_size = 2 * 1024 * 1024; // 2 MB
    nexus::core::DeviceMemoryArena arena(arena_size, 256);

    // Fill arena
    arena.allocate(1024 * 1024);
    arena.allocate(1024 * 1024);
    if (arena.remaining_bytes() != 0) {
        throw std::runtime_error("Remaining bytes should be 0 after filling");
    }

    // Reset
    arena.reset(0);
    if (arena.allocated_bytes() != 0 || arena.remaining_bytes() != arena_size) {
        throw std::runtime_error("Arena reset failed to restore capacity");
    }

    // Reallocate
    void* ptr = arena.allocate(1024 * 1024);
    if (ptr != arena.base_ptr()) {
        throw std::runtime_error("First allocation after reset must return base pointer");
    }
    std::cout << "PASSED\n";
}

void testGpuKernelMultiRegion() {
    std::cout << "[TEST] 4. Multi-Region GPU Kernel Execution & Boundary Integrity... ";
    constexpr size_t num_elements = 4096;
    constexpr size_t region_bytes = num_elements * sizeof(float); // 16 KB each
    constexpr size_t arena_size = 1024 * 1024; // 1 MB

    nexus::core::DeviceMemoryArena arena(arena_size, 256);

    // Sub-allocate 3 distinct tensor buffers from the single arena block
    float* d_a = static_cast<float*>(arena.allocate(region_bytes));
    float* d_b = static_cast<float*>(arena.allocate(region_bytes));
    float* d_c = static_cast<float*>(arena.allocate(region_bytes));

    if (!arena.contains(d_a, region_bytes) || !arena.contains(d_b, region_bytes) || !arena.contains(d_c, region_bytes)) {
        throw std::runtime_error("Sub-allocated buffers must be contained in arena");
    }

    // Prepare host data
    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    std::vector<float> h_c(num_elements, 0.0f);

    for (size_t i = 0; i < num_elements; ++i) {
        h_a[i] = static_cast<float>(i) * 1.5f;
        h_b[i] = static_cast<float>(i) * 2.5f;
    }

    // Copy to sub-allocated arena regions
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), region_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), region_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, region_bytes));

    // Launch kernel reading from arena d_a, d_b and writing into arena d_c
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((num_elements + block_size - 1) / block_size);
    arenaVectorAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, num_elements);
    CUDA_SYNC_AND_CHECK();

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, region_bytes, cudaMemcpyDeviceToHost));

    // Verify
    for (size_t i = 0; i < num_elements; ++i) {
        float expected = h_a[i] + h_b[i];
        float diff = std::fabs(h_c[i] - expected);
        if (diff >= 1e-4f) {
            throw std::runtime_error("Kernel computation on arena buffers produced incorrect result");
        }
    }
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "========================================================\n"
              << "       NEXUS-CUDA MEMORY ARENA TEST SUITE (P1)          \n"
              << "========================================================\n";

    try {
        testAlignment();
        testExhaustion();
        testResetAndReuse();
        testGpuKernelMultiRegion();

        std::cout << "\n[ALL TESTS PASSED] DeviceMemoryArena satisfies all P1 criteria!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED] " << e.what() << "\n";
        return 1;
    }
}
