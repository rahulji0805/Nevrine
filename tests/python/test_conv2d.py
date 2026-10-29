import pytest
import numpy as np
import nevrine


def naive_conv2d_reference(x, w, b=None, stride=1, pad=0):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for co in range(C_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    acc = 0.0
                    for ci in range(C_in):
                        for fh in range(kH):
                            for fw in range(kW):
                                ih = oh * stride + fh - pad
                                iw = ow * stride + fw - pad
                                if 0 <= ih < H and 0 <= iw < W:
                                    acc += x[n, ci, ih, iw] * w[co, ci, fh, fw]
                    if b is not None:
                        acc += b[co]
                    out[n, co, oh, ow] = acc
    return out


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_conv2d_basic_and_pad(backend):
    np.random.seed(42)
    x = np.random.randn(1, 1, 8, 8).astype(np.float32)
    w = np.random.randn(1, 1, 3, 3).astype(np.float32)

    expected = naive_conv2d_reference(x, w, stride=1, pad=1)
    actual = nevrine.conv2d(x, w, stride=1, pad=1, backend=backend)

    assert actual.shape == (1, 1, 8, 8)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_conv2d_multi_channel_bias(backend):
    np.random.seed(123)
    x = np.random.randn(2, 3, 16, 16).astype(np.float32)
    w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)

    expected = naive_conv2d_reference(x, w, b, stride=1, pad=1)
    actual = nevrine.conv2d(x, w, bias=b, stride=1, pad=1, backend=backend)

    assert actual.shape == (2, 4, 16, 16)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_conv2d_fused_relu(backend):
    np.random.seed(456)
    x = np.random.randn(2, 2, 8, 8).astype(np.float32)
    w = np.random.randn(2, 2, 3, 3).astype(np.float32)
    b = np.random.randn(2).astype(np.float32)

    expected = naive_conv2d_reference(x, w, b, stride=1, pad=0)
    expected = np.maximum(0.0, expected)

    actual = nevrine.conv2d(x, w, bias=b, stride=1, pad=0, fuse_relu=True, backend=backend)

    assert (actual >= 0.0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
