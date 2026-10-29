"""Nevrine: Cloud-Native Neural Acceleration Engine Python Package."""

import sys
from typing import Optional, Union, Dict, Any
import numpy as np

try:
    from . import _nevrine
except ImportError as e:
    # Try importing from build or current path
    try:
        import _nevrine
    except ImportError:
        raise ImportError(
            f"Failed to load Nevrine native backend '_nevrine'. Ensure the C++ extension is compiled.\nOriginal error: {e}"
        ) from e

__version__ = "0.5.0"

# Expose C++ Core Classes
DeviceManager = _nevrine.DeviceManager
DeviceMemoryArena = _nevrine.DeviceMemoryArena

# Add context manager support to DeviceMemoryArena for clean Pythonic usage
def _arena_enter(self):
    return self

def _arena_exit(self, exc_type, exc_val, exc_tb):
    self.reset()
    return False

DeviceMemoryArena.__enter__ = _arena_enter
DeviceMemoryArena.__exit__ = _arena_exit


def get_device_count() -> int:
    """Returns the total number of CUDA-capable devices available."""
    return DeviceManager.get_device_count()


def get_current_device() -> int:
    """Returns the ID of the currently active CUDA device."""
    return DeviceManager.get_current_device()


def set_device(device_id: int) -> None:
    """Sets the active CUDA device."""
    DeviceManager.set_device(device_id)


def get_memory_info(device_id: int = 0) -> Dict[str, Any]:
    """Returns VRAM memory metrics (free, used, total in bytes and MiB) for the specified device."""
    return DeviceManager.get_memory_info(device_id)


def get_device_properties(device_id: int = 0) -> Dict[str, Any]:
    """Returns hardware specifications (compute capability, multiprocessors, warp size, etc.)."""
    return DeviceManager.get_device_properties(device_id)


def print_device_summary(device_id: int = 0) -> None:
    """Prints a formatted hardware summary of the device to stdout."""
    DeviceManager.print_device_summary(device_id)


def gemm(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: str = "cuda_tiled",
) -> np.ndarray:
    """Executes General Matrix Multiplication: C = alpha * (A @ B) + beta * C.

    Parameters
    ----------
    a : np.ndarray
        2D matrix of shape (M, K)
    b : np.ndarray
        2D matrix of shape (K, N)
    alpha : float, optional
        Scalar multiplier for (A @ B), by default 1.0
    beta : float, optional
        Scalar multiplier for C, by default 0.0
    backend : str, optional
        Execution backend: 'cuda_tiled' (default), 'cuda_naive', 'cublas', or 'cpu'

    Returns
    -------
    np.ndarray
        Output matrix of shape (M, N) with dtype float32.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Inputs A and B must be numpy ndarrays")

    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if b.dtype != np.float32:
        b = b.astype(np.float32)

    return _nevrine.gemm(a, b, float(alpha), float(beta), str(backend))


def conv2d(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    pad: int = 0,
    fuse_relu: bool = False,
    backend: str = "cuda",
) -> np.ndarray:
    """Executes 2D Convolution forward pass for NCHW tensors.

    Parameters
    ----------
    input : np.ndarray
        4D input tensor of shape (N, C_in, H, W)
    kernel : np.ndarray
        4D kernel tensor of shape (C_out, C_in, kH, kW)
    bias : Optional[np.ndarray], optional
        1D bias tensor of shape (C_out,), by default None
    stride : int, optional
        Convolution spatial stride, by default 1
    pad : int, optional
        Symmetric zero-padding, by default 0
    fuse_relu : bool, optional
        If True, fuses ReLU activation in-kernel to eliminate memory round-trip, by default False
    backend : str, optional
        Execution backend: 'cuda' (default) or 'cpu'

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C_out, H_out, W_out) with dtype float32.
    """
    if not isinstance(input, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise TypeError("Input and kernel must be numpy ndarrays")

    if input.dtype != np.float32:
        input = input.astype(np.float32)
    if kernel.dtype != np.float32:
        kernel = kernel.astype(np.float32)

    if bias is not None:
        if not isinstance(bias, np.ndarray):
            raise TypeError("Bias must be a numpy ndarray if provided")
        if bias.dtype != np.float32:
            bias = bias.astype(np.float32)

    return _nevrine.conv2d(
        input, kernel, bias, int(stride), int(pad), bool(fuse_relu), str(backend)
    )


def relu(
    x: np.ndarray,
    inplace: bool = False,
    backend: str = "cuda",
) -> np.ndarray:
    """Applies elementwise Rectified Linear Unit activation: max(0, x).

    Parameters
    ----------
    x : np.ndarray
        Input array of arbitrary shape
    inplace : bool, optional
        Whether to perform the operation in-place on the input array, by default False
    backend : str, optional
        Execution backend: 'cuda' (default) or 'cpu'

    Returns
    -------
    np.ndarray
        Activated array with dtype float32.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy ndarray")

    if x.dtype != np.float32:
        x = x.astype(np.float32)

    return _nevrine.relu(x, bool(inplace), str(backend))


def leaky_relu(
    x: np.ndarray,
    negative_slope: float = 0.01,
    inplace: bool = False,
    backend: str = "cuda",
) -> np.ndarray:
    """Applies elementwise Leaky ReLU activation: x if x > 0 else negative_slope * x.

    Parameters
    ----------
    x : np.ndarray
        Input array of arbitrary shape
    negative_slope : float, optional
        Slope for negative values, by default 0.01
    inplace : bool, optional
        Whether to perform the operation in-place on the input array, by default False
    backend : str, optional
        Execution backend: 'cuda' (default) or 'cpu'

    Returns
    -------
    np.ndarray
        Activated array with dtype float32.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy ndarray")

    if x.dtype != np.float32:
        x = x.astype(np.float32)

    return _nevrine.leaky_relu(x, float(negative_slope), bool(inplace), str(backend))


__all__ = [
    "DeviceManager",
    "DeviceMemoryArena",
    "get_device_count",
    "get_current_device",
    "set_device",
    "get_memory_info",
    "get_device_properties",
    "print_device_summary",
    "gemm",
    "conv2d",
    "relu",
    "leaky_relu",
]
