import pytest
import numpy as np
import nevrine


@pytest.mark.parametrize("backend", ["cuda_tiled", "cuda_naive", "cublas", "cpu"])
def test_gemm_square(backend):
    np.random.seed(42)
    M, K, N = 64, 64, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_expected = A @ B
    C_actual = nevrine.gemm(A, B, alpha=1.0, beta=0.0, backend=backend)

    assert C_actual.shape == (M, N)
    np.testing.assert_allclose(C_actual, C_expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("backend", ["cuda_tiled", "cuda_naive", "cublas", "cpu"])
def test_gemm_rectangular_and_prime(backend):
    np.random.seed(123)
    # Non-power-of-two / prime shapes
    M, K, N = 17, 31, 43
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_expected = A @ B
    C_actual = nevrine.gemm(A, B, alpha=1.0, beta=0.0, backend=backend)

    assert C_actual.shape == (M, N)
    np.testing.assert_allclose(C_actual, C_expected, rtol=1e-3, atol=1e-4)


def test_gemm_alpha_scaling():
    np.random.seed(99)
    M, K, N = 32, 32, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    alpha = 2.5
    C_expected = alpha * (A @ B)
    C_actual = nevrine.gemm(A, B, alpha=alpha, backend="cuda_tiled")

    np.testing.assert_allclose(C_actual, C_expected, rtol=1e-3, atol=1e-4)


def test_gemm_dimension_mismatch():
    A = np.ones((10, 20), dtype=np.float32)
    B = np.ones((25, 30), dtype=np.float32)

    with pytest.raises(Exception):
        nevrine.gemm(A, B)
