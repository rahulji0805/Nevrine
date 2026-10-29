import pytest
import nevrine


def test_arena_initialization():
    capacity = 1024 * 1024  # 1 MB
    arena = nevrine.DeviceMemoryArena(capacity, 256)
    assert arena.capacity() == capacity
    assert arena.allocated_bytes() == 0
    assert arena.remaining_bytes() == capacity
    assert arena.alignment() == 256
    assert arena.allocation_count() == 0


def test_arena_reset():
    capacity = 2 * 1024 * 1024  # 2 MB
    arena = nevrine.DeviceMemoryArena(capacity, 256)
    arena.reset()
    assert arena.allocated_bytes() == 0
    assert arena.remaining_bytes() == capacity


def test_arena_context_manager():
    capacity = 1024 * 1024
    with nevrine.DeviceMemoryArena(capacity) as arena:
        assert arena.capacity() == capacity
        assert arena.allocated_bytes() == 0
    # After exiting context manager, arena reset is called
    assert arena.allocated_bytes() == 0
