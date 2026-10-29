"""Nevrine Cloud-Native Neural Acceleration FastAPI Serving Application."""

from contextlib import asynccontextmanager
import time
from typing import Dict, List
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

import nevrine
from nevrine.models import ConvNetClassifier
from serving.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    DevicePropertiesResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# Global model instance
model: ConvNetClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms the Nevrine inference engine at startup."""
    global model
    print("[Nevrine Server] Initializing inference engine...")
    device_count = nevrine.get_device_count()
    if device_count > 0:
        props = nevrine.get_device_properties(0)
        print(f"[Nevrine Server] Active GPU: {props['name']} (sm_{props['compute_capability']})")
    else:
        print("[Nevrine Server] No GPU found. Falling back to CPU backend.")

    # Initialize model with 1-channel 28x28 input (e.g. MNIST / FashionMNIST size)
    model = ConvNetClassifier(
        in_channels=1,
        in_height=28,
        in_width=28,
        num_classes=10,
        backend="cuda_tiled" if device_count > 0 else "cpu",
    )

    # Pre-warm GPU context with a dummy forward pass
    dummy_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    _ = model.forward(dummy_input)
    print("[Nevrine Server] Engine pre-warmed successfully. Ready for inference.")

    yield

    print("[Nevrine Server] Shutting down.")


app = FastAPI(
    title="Nevrine Inference Service",
    description="Cloud-Native Low-Latency Neural Acceleration Engine",
    version="0.6.0",
    lifespan=lifespan,
)

# Enable CORS for web dashboard & frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health():
    """Returns engine operational health, device identification, and VRAM utilization."""
    try:
        mem = nevrine.get_memory_info(0)
        props = nevrine.get_device_properties(0)
        return HealthResponse(
            status="ok",
            cuda_device_id=0,
            cuda_device_name=props["name"],
            vram_total_mib=mem["total_mib"],
            vram_free_mib=mem["free_mib"],
            vram_used_mib=mem["used_mib"],
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            cuda_device_id=-1,
            cuda_device_name="CPU Fallback",
            vram_total_mib=0.0,
            vram_free_mib=0.0,
            vram_used_mib=0.0,
        )


@app.get("/device", response_model=DevicePropertiesResponse, tags=["Hardware"])
def get_device():
    """Returns detailed hardware architecture specifications."""
    try:
        props = nevrine.get_device_properties(0)
        return DevicePropertiesResponse(**props)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query device properties: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(req: PredictionRequest):
    """Executes low-latency CNN forward pass on an NCHW batch."""
    global model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not yet initialized",
        )

    try:
        arr = np.array(req.data, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (N, C, H, W). Received shape: {arr.shape}")
        if arr.shape[1] != model.in_channels or arr.shape[2] != model.in_height or arr.shape[3] != model.in_width:
            raise ValueError(
                f"Dimension mismatch: expected (_, {model.in_channels}, {model.in_height}, {model.in_width}), received {arr.shape}"
            )

        # Temporary backend override if requested
        original_backend = model.backend
        if req.backend:
            model.backend = req.backend

        probs, breakdown = model.forward(arr, return_timing=True)
        model.backend = original_backend

        classes = np.argmax(probs, axis=-1).tolist()
        return PredictionResponse(
            batch_size=arr.shape[0],
            predicted_classes=classes,
            probabilities=probs.tolist(),
            inference_time_ms=breakdown["total_ms"],
            breakdown_ms=breakdown,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmarking"])
def benchmark(req: BenchmarkRequest):
    """Executes an in-flight benchmark measuring latency percentiles (p50, p90, p95, p99)."""
    global model
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model uninitialized")

    try:
        backend = req.backend or model.backend
        batch = np.random.randn(req.batch_size, model.in_channels, model.in_height, model.in_width).astype(np.float32)

        original_backend = model.backend
        model.backend = backend

        # Warm-up iterations
        for _ in range(req.warmup):
            _ = model.forward(batch)

        # Measured iterations
        latencies_ms: List[float] = []
        for _ in range(req.iterations):
            t0 = time.perf_counter()
            _ = model.forward(batch)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        model.backend = original_backend

        latencies_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_arr))
        p50 = float(np.percentile(latencies_arr, 50))
        p90 = float(np.percentile(latencies_arr, 90))
        p95 = float(np.percentile(latencies_arr, 95))
        p99 = float(np.percentile(latencies_arr, 99))
        throughput = float(req.batch_size / (mean_lat / 1000.0))

        return BenchmarkResponse(
            backend=backend,
            batch_size=req.batch_size,
            iterations=req.iterations,
            throughput_samples_per_sec=throughput,
            latency_mean_ms=mean_lat,
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            min_ms=float(np.min(latencies_arr)),
            max_ms=float(np.max(latencies_arr)),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.app:app", host="0.0.0.0", port=8000, reload=False)
