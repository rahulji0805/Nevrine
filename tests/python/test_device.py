import nevrine


def test_device_count():
    count = nevrine.get_device_count()
    assert count >= 1, f"Expected at least 1 CUDA device, got {count}"


def test_device_properties():
    props = nevrine.get_device_properties(0)
    assert "name" in props
    assert "compute_capability" in props
    assert props["total_global_mem_bytes"] > 0
    assert props["multi_processor_count"] > 0
    assert props["warp_size"] == 32


def test_memory_info():
    mem = nevrine.get_memory_info(0)
    assert mem["total_bytes"] > 0
    assert mem["free_bytes"] > 0
    assert mem["total_bytes"] >= mem["free_bytes"]
    assert mem["total_mib"] > 0
