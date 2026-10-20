#pragma once

#include "nevrine/core/cuda_utils.hpp"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nevrine {
namespace core {

/**
 * @brief Exception thrown when an arena allocation exceeds the reserved capacity.
 */
class ArenaOutOfMemoryException : public std::runtime_error {
public:
    ArenaOutOfMemoryException(size_t requested_bytes, size_t available_bytes, size_t capacity_bytes)
        : std::runtime_error(formatMessage(requested_bytes, available_bytes, capacity_bytes)),
          requested_bytes_(requested_bytes),
          available_bytes_(available_bytes),
          capacity_bytes_(capacity_bytes) {}

    size_t requested_bytes() const noexcept { return requested_bytes_; }
    size_t available_bytes() const noexcept { return available_bytes_; }
    size_t capacity_bytes() const noexcept { return capacity_bytes_; }

private:
    static std::string formatMessage(size_t req, size_t avail, size_t cap) {
        std::ostringstream oss;
        oss << "[ARENA OUT OF MEMORY] Failed to allocate " << req << " bytes. "
            << "Available: " << avail << " bytes, Total Capacity: " << cap << " bytes.";
        return oss.str();
    }

    size_t requested_bytes_;
    size_t available_bytes_;
    size_t capacity_bytes_;
};

/**
 * @brief Handle representing a sub-allocation inside a DeviceMemoryArena.
 */
struct ArenaAllocation {
    void* ptr{nullptr};
    size_t offset{0};
    size_t size_bytes{0};
    size_t aligned_size_bytes{0};
};

/**
 * @brief Monolithic, fixed-capacity device memory arena with 256-byte aligned bump allocations.
 *
 * Guarantees zero steady-state `cudaMalloc` / `cudaFree` calls during execution.
 */
class DeviceMemoryArena {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 256; // 256 bytes for optimal memory coalescing

    /**
     * @brief Allocates a monolithic device VRAM block.
     * @param capacity_bytes Total memory in bytes to reserve.
     * @param alignment Byte alignment for each sub-allocation (default 256).
     */
    explicit DeviceMemoryArena(size_t capacity_bytes, size_t alignment = DEFAULT_ALIGNMENT);

    ~DeviceMemoryArena();

    // Non-copyable
    DeviceMemoryArena(const DeviceMemoryArena&) = delete;
    DeviceMemoryArena& operator=(const DeviceMemoryArena&) = delete;

    // Movable
    DeviceMemoryArena(DeviceMemoryArena&& other) noexcept;
    DeviceMemoryArena& operator=(DeviceMemoryArena&& other) noexcept;

    /**
     * @brief Sub-allocates an aligned region of device memory from the arena.
     * @param size_bytes Number of bytes requested.
     * @return Raw device pointer to the allocated region.
     * @throws ArenaOutOfMemoryException if requested size exceeds remaining capacity.
     */
    void* allocate(size_t size_bytes);

    /**
     * @brief Sub-allocates and returns full metadata handle.
     */
    ArenaAllocation allocateHandle(size_t size_bytes);

    /**
     * @brief Resets the allocation offset back to zero (or specified watermark).
     *        Does NOT free the underlying device memory block.
     */
    void reset(size_t watermark = 0);

    /**
     * @brief Verifies whether a given device pointer and byte range is entirely contained in this arena.
     */
    bool contains(const void* ptr, size_t size_bytes = 0) const noexcept;

    // Accessors
    size_t capacity() const noexcept { return capacity_bytes_; }
    size_t allocated_bytes() const noexcept { return current_offset_; }
    size_t remaining_bytes() const noexcept { return capacity_bytes_ - current_offset_; }
    size_t alignment() const noexcept { return alignment_; }
    size_t allocation_count() const noexcept { return allocation_count_; }
    void* base_ptr() const noexcept { return base_ptr_; }

private:
    static size_t alignUp(size_t offset, size_t alignment) noexcept {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    void* base_ptr_{nullptr};
    size_t capacity_bytes_{0};
    size_t alignment_{DEFAULT_ALIGNMENT};
    size_t current_offset_{0};
    size_t allocation_count_{0};
};

} // namespace core
} // namespace nevrine
