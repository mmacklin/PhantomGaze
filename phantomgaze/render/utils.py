# Helpful functions

import cupy as cp
import numba
from numba import cuda

@cuda.jit(device=True)
def _safe_index_array(
        array,
        i,
        j,
        k):
    """Get an index of an array, clamping to the array bounds.

    Parameters
    ----------
    array : ndarray
        The array to index.
    i : int
        The i index.
    j : int
        The j index.
    k : int
        The k index.
    """

    # Clamp the index to the array
    if i < 0 or i >= array.shape[0]:
        i = min(max(i, 0), array.shape[0] - 1)
    if j < 0 or j >= array.shape[1]:
        j = min(max(j, 0), array.shape[1] - 1)
    if k < 0 or k >= array.shape[2]:
        k = min(max(k, 0), array.shape[2] - 1)

    # Return the value at the index
    return array[i, j, k]

@cuda.jit(device=True)
def _trilinear_interpolation(
        v_000,
        v_100,
        v_010,
        v_110,
        v_001,
        v_101,
        v_011,
        v_111,
        dx,
        dy,
        dz):
    """Perform trilinear interpolation on 8 points.

    Parameters
    ----------
    v_000 : float
    v_100 : float
    v_010 : float
    v_110 : float
    v_001 : float
    v_101 : float
    v_011 : float
    v_111 : float
    dx : float
    dy : float
    dz : float
    """

    # Compute the interpolation
    v_00 = v_000 * (1 - dx) + v_100 * dx
    v_10 = v_010 * (1 - dx) + v_110 * dx
    v_01 = v_001 * (1 - dx) + v_101 * dx
    v_11 = v_011 * (1 - dx) + v_111 * dx
    v_0 = v_00 * (1 - dy) + v_10 * dy
    v_1 = v_01 * (1 - dy) + v_11 * dy
    v = v_0 * (1 - dz) + v_1 * dz

    # Return the interpolated value
    return v

@cuda.jit(device=True)
def sample_array(
        array,
        spacing,
        origin,
        position):
    """Sample an array at a given position.
    Uses trilinear interpolation.

    Parameters
    ----------
    array : ndarray
        The volume data.
    spacing : tuple
        The spacing of the volume data.
    origin : tuple
        The origin of the volume data.
    position : tuple
        The position to sample.
    """

    # Get the lower i, j, and k indices of the volume
    i = int((position[0] - origin[0]) / spacing[0])
    j = int((position[1] - origin[1]) / spacing[1])
    k = int((position[2] - origin[2]) / spacing[2])

    # Get the fractional part of the indices
    dx = (position[0] - origin[0]) / spacing[0] - i
    dy = (position[1] - origin[1]) / spacing[1] - j
    dz = (position[2] - origin[2]) / spacing[2] - k

    # Sample the array at the indices
    v_000 = _safe_index_array(array, i, j, k)
    v_100 = _safe_index_array(array, i + 1, j, k)
    v_010 = _safe_index_array(array, i, j + 1, k)
    v_110 = _safe_index_array(array, i + 1, j + 1, k)
    v_001 = _safe_index_array(array, i, j, k + 1)
    v_101 = _safe_index_array(array, i + 1, j, k + 1)
    v_011 = _safe_index_array(array, i, j + 1, k + 1)
    v_111 = _safe_index_array(array, i + 1, j + 1, k + 1)

    # Perform trilinear interpolation
    return _trilinear_interpolation(
        v_000,
        v_100,
        v_010,
        v_110,
        v_001,
        v_101,
        v_011,
        v_111,
        dx,
        dy,
        dz)



@cuda.jit(device=True)
def sample_array_derivative(
        array,
        spacing,
        origin,
        position):
    """Compute the derivative of an array at a given position.

    Parameters
    ----------
    array : ndarray
        The volume data.
    spacing : tuple
        The spacing of the volume data.
    origin : tuple
        The origin of the volume data.
    position : tuple
        The position to sample.
    """
     
    # Get the i, j, and k indices of the volume
    i = int((position[0] - origin[0]) / spacing[0])
    j = int((position[1] - origin[1]) / spacing[1])
    k = int((position[2] - origin[2]) / spacing[2])

    # If the indices are out of bounds set the derivative to be normal of the array
    if i <= 0:
        return (-1, 0, 0)
    elif i >= array.shape[0]-1:
        return (1, 0, 0)
    elif j <= 0:
        return (0, -1, 0)
    elif j >= array.shape[1]-1:
        return (0, 1, 0)
    elif k <= 0:
        return (0, 0, -1)
    elif k >= array.shape[2]-1:
        return (0, 0, 1)

    # Compute the derivative
    array_dx = (array[i + 1, j, k] - array[i - 1, j, k]) / (2 * spacing[0])
    array_dy = (array[i, j + 1, k] - array[i, j - 1, k]) / (2 * spacing[1])
    array_dz = (array[i, j, k + 1] - array[i, j, k - 1]) / (2 * spacing[2])

    # Return the derivative
    return (array_dx, array_dy, array_dz)


