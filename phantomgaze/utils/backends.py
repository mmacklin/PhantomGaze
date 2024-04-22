import logging
import cupy as cp
import numpy as np

# Import backends
# jax
try:
    import jax.dlpack as jdlpack
    import jax.numpy as jnp
except ImportError:
    logging.warning("JAX not installed, JAX backend not available")

# warp
try:
    import warp as wp
except ImportError:
    logging.warning("Warp not installed, Warp backend not available")

# torch
try:
    import torch
except ImportError:
    logging.warning("Torch not installed, Torch backend not available")


def backend_to_cupy(backend_array):
    """
    Convert backend array to cupy array

    Parameters
    ----------
    backend_array : array
        Array from backend, can be jax, warp, or torch
    """

    # Perform zero-copy conversion to cupy array
    if isinstance(backend_array, cp.ndarray):
        return backend_array
    elif isinstance(backend_array, jnp.ndarray):
        dl_array = jdlpack.to_dlpack(backend_array)
        cupy_array = cp.fromDlpack(dl_array)
    elif isinstance(backend_array, wp.array):
        dl_array = wp.to_dlpack(backend_array)
        cupy_array = cp.fromDlpack(dl_array)
    elif isinstance(backend_array, torch.Tensor):
        cupy_array = cp.fromDlpack(backend_array.toDlpack())
    else:
        raise ValueError(f"Backend {type(backend_array)} not supported")
    return cupy_array


def to_warp(input_array):
    """
    Convert input array to Warp array

    Parameters
    ----------
    input_array : array
        Array from input, can be JAX, NumPy, Warp, or Torch
    """

    # Perform zero-copy conversion to cupy array
    if isinstance(input_array, wp.array):
        return input_array
    elif isinstance(input_array, cp.ndarray):
        return wp.array(input_array.get(), dtype=wp.float32) # TODO: how to use __cuda_array_interface__ instead of going through NumPy (.get()) here?
    elif isinstance(input_array, jnp.ndarray):
        return wp.from_jax(input_array)
    elif isinstance(input_array, np.ndarray):
        return wp.from_numpy(input_array)
    elif isinstance(input_array, torch.Tensor):
        return wp.from_torch(input_array)
    else:
        raise ValueError(f"Input {type(input_array)} not supported")
