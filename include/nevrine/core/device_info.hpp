#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace nevrine {
namespace core {

struct MemoryInfo {
    size_t free_bytes{0};
    size_t total_bytes{0};
    size_t used_bytes{0};

    double free_mib() const { return static_cast<double>(free_bytes) / (1024.0 * 1024.0); }
    double total_mib() const { return static_cast<double>(total_bytes) / (1024.0 * 1024.0); }
    double used_mib() const { return static_cast<double>(used_bytes) / (1024.0 * 1024.0); }
};

struct DeviceProperties {
    int device_id{0};
    std::string name;
    int major{0};
    int minor{0};
    size_t total_global_mem{0};
    int multi_processor_count{0};
    int max_threads_per_block{0};
    int warp_size{0};
    size_t shared_mem_per_block{0};
    int l2_cache_size{0};
    int memory_clock_rate_khz{0};
    int memory_bus_width_bits{0};
    bool can_map_host_memory{false};
    bool unified_addressing{false};
};

class DeviceManager {
public:
    static int getDeviceCount();
    static void setDevice(int device_id);
    static int getCurrentDevice();
    static DeviceProperties getDeviceProperties(int device_id = 0);
    static MemoryInfo getMemoryInfo();
    static void printDeviceSummary(int device_id = 0);
};

} // namespace core
} // namespace nevrine
