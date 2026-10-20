#include "nexus/core/cuda_utils.hpp"
#include "nexus/core/device_info.hpp"
#include "nexus/cuda/smoke_kernel.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Starting NEXUS-CUDA Environment Verification (Session 1 / P0)...\n\n";

    try {
        int device_count = nexus::core::DeviceManager::getDeviceCount();
        if (device_count == 0) {
            std::cerr << "[ERROR] No CUDA-capable devices found!\n";
            return EXIT_FAILURE;
        }

        std::cout << "Discovered " << device_count << " CUDA device(s).\n\n";

        // Query properties for primary device (device 0)
        nexus::core::DeviceManager::setDevice(0);
        nexus::core::DeviceManager::printDeviceSummary(0);

        std::cout << "\nExecuting CUDA Smoke Test Kernel...\n";
        constexpr size_t test_elements = 65536; // 64K elements
        bool passed = nexus::cuda::runSmokeTest(test_elements);

        if (!passed) {
            std::cerr << "\n[RESULT] Environment verification FAILED: smoke test mismatch.\n";
            return EXIT_FAILURE;
        }

        std::cout << "\n[RESULT] Environment verification SUCCESSFUL: P0 milestone passed!\n";
        return EXIT_SUCCESS;

    } catch (const nexus::core::CudaException& e) {
        std::cerr << "\n" << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "\n[STANDARD EXCEPTION] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
