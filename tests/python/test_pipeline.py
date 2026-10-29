import os
import tempfile
import pytest
import numpy as np

from nevrine.models import ConvNetClassifier, max_pool2d, softmax


def test_max_pool2d():
    x = np.array([[[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0]]]], dtype=np.float32)
    # Shape: (1, 1, 4, 4) -> (1, 1, 2, 2)
    pooled = max_pool2d(x, pool_size=2, stride=2)
    expected = np.array([[[[6.0, 8.0],
                          [9.0, 7.0]]]], dtype=np.float32)
    np.testing.assert_allclose(pooled, expected)


def test_softmax_properties():
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]], dtype=np.float32)
    probs = softmax(x, axis=-1)

    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), np.array([1.0, 1.0]), atol=1e-5)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


@pytest.mark.parametrize("backend", ["cuda_tiled", "cuda_naive", "cublas", "cpu"])
def test_convnet_forward_backends(backend):
    np.random.seed(42)
    model = ConvNetClassifier(
        in_channels=1,
        in_height=28,
        in_width=28,
        num_classes=10,
        conv1_channels=8,
        conv2_channels=16,
        fc_hidden=64,
        backend=backend,
    )

    batch_size = 2
    x = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)

    probs, timings = model.forward(x, return_timing=True)

    assert probs.shape == (batch_size, 10)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(batch_size), atol=1e-4)

    assert "total_ms" in timings
    assert timings["total_ms"] > 0
    assert "conv1_relu_ms" in timings
    assert "fc1_relu_ms" in timings


def test_convnet_predict():
    model = ConvNetClassifier(in_channels=1, in_height=28, in_width=28, num_classes=5)
    x = np.random.randn(3, 1, 28, 28).astype(np.float32)
    preds = model.predict(x)

    assert preds.shape == (3,)
    assert (preds >= 0).all() and (preds < 5).all()


def test_convnet_weights_save_load():
    model1 = ConvNetClassifier(in_channels=1, in_height=28, in_width=28, num_classes=10)
    x = np.random.randn(1, 1, 28, 28).astype(np.float32)
    out1 = model1.forward(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        weight_path = os.path.join(tmpdir, "model_weights.npz")
        model1.save_weights(weight_path)

        model2 = ConvNetClassifier(in_channels=1, in_height=28, in_width=28, num_classes=10)
        # Randomly initialize differently
        model2.init_weights(seed=999)
        model2.load_weights(weight_path)

        out2 = model2.forward(x)
        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)
