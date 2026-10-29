import pytest
import numpy as np
from fastapi.testclient import TestClient

from serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "cuda_device_name" in data
    assert "vram_total_mib" in data


def test_device_endpoint(client):
    response = client.get("/device")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "compute_capability" in data
    assert "warp_size" in data


def test_predict_endpoint(client):
    # Create valid 1x1x28x28 sample image
    dummy_image = np.random.randn(1, 1, 28, 28).astype(np.float32).tolist()
    payload = {
        "data": dummy_image,
        "backend": "cuda_tiled",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 1
    assert len(data["predicted_classes"]) == 1
    assert len(data["probabilities"][0]) == 10
    assert data["inference_time_ms"] > 0
    assert "breakdown_ms" in data


def test_predict_batch_cpu_fallback(client):
    # Batch size 2 on CPU backend
    dummy_image = np.random.randn(2, 1, 28, 28).astype(np.float32).tolist()
    payload = {
        "data": dummy_image,
        "backend": "cpu",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predicted_classes"]) == 2


def test_predict_invalid_shape(client):
    # 2D instead of 4D tensor
    invalid_data = [[1.0, 2.0], [3.0, 4.0]]
    response = client.post("/predict", json={"data": invalid_data})
    assert response.status_code in [400, 422]


def test_benchmark_endpoint(client):
    payload = {
        "iterations": 10,
        "warmup": 3,
        "batch_size": 1,
        "backend": "cuda_tiled",
    }
    response = client.post("/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["iterations"] == 10
    assert data["throughput_samples_per_sec"] > 0
    assert data["latency_p50_ms"] > 0
    assert data["latency_p95_ms"] >= data["latency_p50_ms"]
