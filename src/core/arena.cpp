#include "nexus/core/arena.hpp"
#include <iostream>
#include <utility>

namespace nexus {
namespace core {

DeviceMemoryArena::DeviceMemoryArena(size_t capacity_bytes, size_t alignment)
    : capacity_bytes_(capacity_bytes), alignment_(alignment), current_offset_(0), allocation_count_(0) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceMemoryArena capacity must be greater than 0");
    }
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        throw std::invalid_argument("DeviceMemoryArena alignment must be a power of two");
    }

    CUDA_CHECK(cudaMalloc(&base_ptr_, capacity_bytes_));
}

DeviceMemoryArena::~DeviceMemoryArena() {
    if (base_ptr_ != nullptr) {
        cudaFree(base_ptr_);
        base_ptr_ = nullptr;
    }
}

DeviceMemoryArena::DeviceMemoryArena(DeviceMemoryArena&& other) noexcept
    : base_ptr_(other.base_ptr_),
      capacity_bytes_(other.capacity_bytes_),
      alignment_(other.alignment_),
      current_offset_(other.current_offset_),
      allocation_count_(other.allocation_count_) {
    other.base_ptr_ = nullptr;
    other.capacity_bytes_ = 0;
    other.current_offset_ = 0;
    other.allocation_count_ = 0;
}

DeviceMemoryArena& DeviceMemoryArena::operator=(DeviceMemoryArena&& other) noexcept {
    if (this != &other) {
        if (base_ptr_ != nullptr) {
            cudaFree(base_ptr_);
        }
        base_ptr_ = other.base_ptr_;
        capacity_bytes_ = other.capacity_bytes_;
        alignment_ = other.alignment_;
        current_offset_ = other.current_offset_;
        allocation_count_ = other.allocation_count_;

        other.base_ptr_ = nullptr;
        other.capacity_bytes_ = 0;
        other.current_offset_ = 0;
        other.allocation_count_ = 0;
    }
    return *this;
}

void* DeviceMemoryArena::allocate(size_t size_bytes) {
    ArenaAllocation handle = allocateHandle(size_bytes);
    return handle.ptr;
}

ArenaAllocation DeviceMemoryArena::allocateHandle(size_t size_bytes) {
    if (size_bytes == 0) {
        throw std::invalid_argument("Cannot allocate 0 bytes from DeviceMemoryArena");
    }

    size_t aligned_start = alignUp(current_offset_, alignment_);
    size_t new_offset = aligned_start + size_bytes;

    if (new_offset > capacity_bytes_) {
        size_t available = (capacity_bytes_ > aligned_start) ? (capacity_bytes_ - aligned_start) : 0;
        throw ArenaOutOfMemoryException(size_bytes, available, capacity_bytes_);
    }

    void* allocated_ptr = static_cast<char*>(base_ptr_) + aligned_start;
    current_offset_ = new_offset;
    allocation_count_++;

    ArenaAllocation handle;
    handle.ptr = allocated_ptr;
    handle.offset = aligned_start;
    handle.size_bytes = size_bytes;
    handle.aligned_size_bytes = new_offset - aligned_start;
    return handle;
}

void DeviceMemoryArena::reset(size_t watermark) {
    if (watermark > capacity_bytes_) {
        throw std::invalid_argument("Reset watermark exceeds arena capacity");
    }
    current_offset_ = watermark;
    if (watermark == 0) {
        allocation_count_ = 0;
    }
}

bool DeviceMemoryArena::contains(const void* ptr, size_t size_bytes) const noexcept {
    if (base_ptr_ == nullptr || ptr == nullptr) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr_);
    const uintptr_t target = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = base + capacity_bytes_;

    return (target >= base) && (target + size_bytes <= end);
}

} // namespace core
} // namespace nexus
