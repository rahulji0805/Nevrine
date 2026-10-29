"""Pydantic schemas for Nevrine FastAPI serving."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    engine: str = Field("Nevrine Neural Acceleration Engine", description="Engine brand")
    version: str = Field("0.6.0", description="Engine version")
    cuda_device_id: int
    cuda_device_name: str
    vram_total_mib: float
    vram_free_mib: float
    vram_used_mib: float


class DevicePropertiesResponse(BaseModel):
    device_id: int
    name: str
    compute_capability: str
    total_global_mem_bytes: int
    multi_processor_count: int
    max_threads_per_block: int
    warp_size: int
    shared_mem_per_block_bytes: int
    l2_cache_size_bytes: int


class PredictionRequest(BaseModel):
    # Tensor can be sent as raw nested list (N, C, H, W)
    data: List[List[List[List[float]]]] = Field(
        ...,
        description="4D NCHW input tensor with shape (N, C, H, W)",
    )
    backend: Optional[str] = Field(
        "cuda_tiled",
        description="Execution backend ('cuda_tiled', 'cuda_naive', 'cublas', or 'cpu')",
    )


class PredictionResponse(BaseModel):
    batch_size: int
    predicted_classes: List[int]
    probabilities: List[List[float]]
    inference_time_ms: float
    breakdown_ms: Dict[str, float]


class BenchmarkRequest(BaseModel):
    iterations: int = Field(50, ge=5, le=500, description="Number of measured forward passes")
    warmup: int = Field(10, ge=1, le=50, description="Number of warm-up iterations")
    batch_size: int = Field(1, ge=1, le=64, description="Inference batch size")
    backend: Optional[str] = Field("cuda_tiled", description="Execution backend")


class BenchmarkResponse(BaseModel):
    backend: str
    batch_size: int
    iterations: int
    throughput_samples_per_sec: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    min_ms: float
    max_ms: float
