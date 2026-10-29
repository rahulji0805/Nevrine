"""Nevrine End-to-End Convolutional Neural Network Classifier."""

import time
from typing import Dict, Optional, Tuple, Union
import numpy as np

import nevrine


def max_pool2d(x: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """Applies 2D max pooling over an NCHW input tensor.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C, H, W)
    pool_size : int, optional
        Window size, by default 2
    stride : int, optional
        Stride, by default 2

    Returns
    -------
    np.ndarray
        Pooled tensor of shape (N, C, H_out, W_out)
    """
    N, C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    # Reshape and take max over pooling windows
    out = np.empty((N, C, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            out[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes numerically stable softmax along the specified axis."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ConvNetClassifier:
    """End-to-End Deep Convolutional Network for Classification.

    Pipeline:
      1. Conv2D (in_channels -> conv1_out, 3x3, pad=1) + Fused ReLU
      2. MaxPool2D (2x2, stride=2)
      3. Conv2D (conv1_out -> conv2_out, 3x3, pad=1) + Fused ReLU
      4. MaxPool2D (2x2, stride=2)
      5. Flatten
      6. Linear (fc1) + ReLU
      7. Linear (fc2) -> logits
      8. Softmax -> probabilities
    """

    def __init__(
        self,
        in_channels: int = 1,
        in_height: int = 28,
        in_width: int = 28,
        num_classes: int = 10,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_hidden: int = 128,
        backend: str = "cuda_tiled",
    ):
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.num_classes = num_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.fc_hidden = fc_hidden
        self.backend = backend

        # Spatial dimensions after 2 pooling stages (downscale by 2 twice)
        self.pooled_h = in_height // 4
        self.pooled_w = in_width // 4
        self.flatten_dim = conv2_channels * self.pooled_h * self.pooled_w

        self.weights: Dict[str, np.ndarray] = {}
        self.init_weights()

    def init_weights(self, seed: int = 42) -> None:
        """Initializes model weights using He (Kaiming) normal initialization."""
        rng = np.random.default_rng(seed)

        # Conv1: (conv1_channels, in_channels, 3, 3)
        std_c1 = np.sqrt(2.0 / (self.in_channels * 9))
        self.weights["c1_w"] = rng.normal(0, std_c1, (self.conv1_channels, self.in_channels, 3, 3)).astype(np.float32)
        self.weights["c1_b"] = np.zeros(self.conv1_channels, dtype=np.float32)

        # Conv2: (conv2_channels, conv1_channels, 3, 3)
        std_c2 = np.sqrt(2.0 / (self.conv1_channels * 9))
        self.weights["c2_w"] = rng.normal(0, std_c2, (self.conv2_channels, self.conv1_channels, 3, 3)).astype(np.float32)
        self.weights["c2_b"] = np.zeros(self.conv2_channels, dtype=np.float32)

        # FC1: (flatten_dim, fc_hidden)
        std_fc1 = np.sqrt(2.0 / self.flatten_dim)
        self.weights["fc1_w"] = rng.normal(0, std_fc1, (self.flatten_dim, self.fc_hidden)).astype(np.float32)
        self.weights["fc1_b"] = np.zeros(self.fc_hidden, dtype=np.float32)

        # FC2: (fc_hidden, num_classes)
        std_fc2 = np.sqrt(2.0 / self.fc_hidden)
        self.weights["fc2_w"] = rng.normal(0, std_fc2, (self.fc_hidden, self.num_classes)).astype(np.float32)
        self.weights["fc2_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray, return_timing: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Runs end-to-end inference pass.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (N, C_in, H, W)
        return_timing : bool, optional
            If True, returns a tuple of (probabilities, timing_dict)

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]
            Predicted class probabilities (N, num_classes), and optional breakdown timings in ms.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D NCHW input tensor, got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {x.shape[1]}")

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # Determine backend for conv2d and gemm
        conv_backend = "cpu" if self.backend == "cpu" else "cuda"
        gemm_backend = self.backend

        # 1. Conv1 + Fused ReLU
        t_start = time.perf_counter()
        x1 = nevrine.conv2d(
            x,
            self.weights["c1_w"],
            bias=self.weights["c1_b"],
            stride=1,
            pad=1,
            fuse_relu=True,
            backend=conv_backend,
        )
        timings["conv1_relu_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 2. MaxPool1
        t_start = time.perf_counter()
        p1 = max_pool2d(x1, pool_size=2, stride=2)
        timings["pool1_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 3. Conv2 + Fused ReLU
        t_start = time.perf_counter()
        x2 = nevrine.conv2d(
            p1,
            self.weights["c2_w"],
            bias=self.weights["c2_b"],
            stride=1,
            pad=1,
            fuse_relu=True,
            backend=conv_backend,
        )
        timings["conv2_relu_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 4. MaxPool2
        t_start = time.perf_counter()
        p2 = max_pool2d(x2, pool_size=2, stride=2)
        timings["pool2_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 5. Flatten: (N, C, H, W) -> (N, C*H*W)
        batch_size = x.shape[0]
        flattened = p2.reshape(batch_size, -1)

        # 6. FC1 (GEMM: (N, flatten_dim) @ (flatten_dim, fc_hidden)) + Bias + ReLU
        t_start = time.perf_counter()
        fc1_out = nevrine.gemm(flattened, self.weights["fc1_w"], alpha=1.0, beta=0.0, backend=gemm_backend)
        fc1_out += self.weights["fc1_b"]
        fc1_act = nevrine.relu(fc1_out, inplace=True, backend=conv_backend)
        timings["fc1_relu_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 7. FC2 (GEMM: (N, fc_hidden) @ (fc_hidden, num_classes)) + Bias
        t_start = time.perf_counter()
        logits = nevrine.gemm(fc1_act, self.weights["fc2_w"], alpha=1.0, beta=0.0, backend=gemm_backend)
        logits += self.weights["fc2_b"]
        timings["fc2_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 8. Softmax
        t_start = time.perf_counter()
        probs = softmax(logits, axis=-1)
        timings["softmax_ms"] = (time.perf_counter() - t_start) * 1000.0

        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

        if return_timing:
            return probs, timings
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns the predicted class indices for the input batch."""
        probs = self.forward(x, return_timing=False)
        return np.argmax(probs, axis=-1)

    def save_weights(self, filepath: str) -> None:
        """Saves weights to a numpy npz archive."""
        np.savez(filepath, **self.weights)

    def load_weights(self, filepath: str) -> None:
        """Loads weights from a numpy npz archive."""
        loaded = np.load(filepath)
        for k in self.weights:
            if k in loaded:
                self.weights[k] = loaded[k].astype(np.float32)
