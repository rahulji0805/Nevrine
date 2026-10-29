import pytest
import numpy as np
import nevrine


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_relu_basic(backend):
    x = np.array([-3.0, -1.0, 0.0, 1.0, 5.0, -10.0], dtype=np.float32)
    expected = np.maximum(0.0, x)

    out = nevrine.relu(x, inplace=False, backend=backend)
    np.testing.assert_allclose(out, expected)
    # Original should be untouched when inplace=False
    assert x[0] == -3.0


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_relu_inplace(backend):
    x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    expected = np.maximum(0.0, x)

    out = nevrine.relu(x, inplace=True, backend=backend)
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(x, expected)


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_leaky_relu(backend):
    x = np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float32)
    slope = 0.1
    expected = np.where(x > 0, x, slope * x)

    out = nevrine.leaky_relu(x, negative_slope=slope, inplace=False, backend=backend)
    np.testing.assert_allclose(out, expected)


def test_activations_large_array():
    x = np.linspace(-100, 100, 100_000, dtype=np.float32)
    expected = np.maximum(0.0, x)

    out_cuda = nevrine.relu(x, backend="cuda")
    out_cpu = nevrine.relu(x, backend="cpu")

    np.testing.assert_allclose(out_cuda, expected)
    np.testing.assert_allclose(out_cpu, expected)
