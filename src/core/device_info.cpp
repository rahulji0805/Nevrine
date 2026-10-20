#include "nevrine/core/device_info.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include <iomanip>
#include <iostream>

namespace nevrine {
namespace core {

int DeviceManager::getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void DeviceManager::setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

int DeviceManager::getCurrentDevice() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
}

DeviceProperties DeviceManager::getDeviceProperties(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceProperties info;
    info.device_id = device_id;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.total_global_mem = prop.totalGlobalMem;
    info.multi_processor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.shared_mem_per_block = prop.sharedMemPerBlock;
    info.l2_cache_size = prop.l2CacheSize;
    info.memory_clock_rate_khz = prop.memoryClockRate;
    info.memory_bus_width_bits = prop.memoryBusWidth;
    info.can_map_host_memory = (prop.canMapHostMemory != 0);
    info.unified_addressing = (prop.unifiedAddressing != 0);

    return info;
}

MemoryInfo DeviceManager::getMemoryInfo() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    MemoryInfo mem;
    mem.free_bytes = free_bytes;
    mem.total_bytes = total_bytes;
    mem.used_bytes = total_bytes - free_bytes;
    return mem;
}

void DeviceManager::printDeviceSummary(int device_id) {
    DeviceProperties prop = getDeviceProperties(device_id);
    MemoryInfo mem = getMemoryInfo();

    std::cout << "========================================================\n"
              << "            Nevrine DEVICE ENVIRONMENT REPORT         \n"
              << "========================================================\n";
    std::cout << std::left << std::setw(28) << "Device ID:" << prop.device_id << "\n";
    std::cout << std::left << std::setw(28) << "Device Name:" << prop.name << "\n";
    std::cout << std::left << std::setw(28) << "Compute Capability:" << prop.major << "." << prop.minor;
    if (prop.major == 8 && prop.minor == 9) {
        std::cout << " (Ada Lovelace Architecture)";
    } else if (prop.major == 7 && prop.minor == 5) {
        std::cout << " (Turing Architecture)";
    }
    std::cout << "\n";
    std::cout << std::left << std::setw(28) << "Streaming Multiprocessors:" << prop.multi_processor_count << "\n";
    std::cout << std::left << std::setw(28) << "Warp Size:" << prop.warp_size << " threads\n";
    std::cout << std::left << std::setw(28) << "Max Threads Per Block:" << prop.max_threads_per_block << "\n";
    std::cout << std::left << std::setw(28) << "Shared Memory / Block:" << (prop.shared_mem_per_block / 1024.0) << " KiB\n";
    std::cout << std::left << std::setw(28) << "L2 Cache Size:" << (prop.l2_cache_size / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << std::left << std::setw(28) << "Memory Bus Width:" << prop.memory_bus_width_bits << " bits\n";
    std::cout << std::left << std::setw(28) << "Total Device Memory:" << std::fixed << std::setprecision(2)
              << (prop.total_global_mem / (1024.0 * 1024.0)) << " MiB ("
              << (prop.total_global_mem / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << std::left << std::setw(28) << "Free VRAM at start:" << mem.free_mib() << " MiB\n";
    std::cout << std::left << std::setw(28) << "Used VRAM at start:" << mem.used_mib() << " MiB\n";
    std::cout << "========================================================\n" << std::flush;
}

} // namespace core
} // namespace nevrine
