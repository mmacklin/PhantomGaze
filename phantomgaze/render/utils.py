# Helpful functions
# TODO: Find better place for these functions

import cupy as cp
import warp as wp

@wp.func
def _safe_index_array(
        array: wp.array3d(dtype=float),
        i: int,
        j: int,
        k: int):
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

@wp.func
def _trilinear_interpolation(
        v_000: float,
        v_100: float,
        v_010: float,
        v_110: float,
        v_001: float,
        v_101: float,
        v_011: float,
        v_111: float,
        dx: float,
        dy: float,
        dz: float):
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
    v_00 = v_000 * (1.0 - dx) + v_100 * dx
    v_10 = v_010 * (1.0 - dx) + v_110 * dx
    v_01 = v_001 * (1.0 - dx) + v_101 * dx
    v_11 = v_011 * (1.0 - dx) + v_111 * dx
    v_0 = v_00 * (1.0 - dy) + v_10 * dy
    v_1 = v_01 * (1.0 - dy) + v_11 * dy
    v = v_0 * (1.0 - dz) + v_1 * dz

    # Return the interpolated value
    return v

@wp.func
def sample_array(
        array: wp.array3d(dtype=float),
        spacing: wp.vec3,
        origin: wp.vec3,
        position: wp.vec3):
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
    dx = (position[0] - origin[0]) / spacing[0] - float(i)
    dy = (position[1] - origin[1]) / spacing[1] - float(j)
    dz = (position[2] - origin[2]) / spacing[2] - float(k)

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

@wp.func
def sample_array_derivative(
        array: wp.array3d(dtype=float),
        spacing: wp.vec3,
        origin: wp.vec3,
        position: wp.vec3):
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

    # Move the position by a small amount
    value_0_1_1 = sample_array(array, spacing, origin, wp.vec3(position[0] - spacing[0]/2.0, position[1], position[2]))
    value_1_0_1 = sample_array(array, spacing, origin, wp.vec3(position[0], position[1] - spacing[1]/2.0, position[2]))
    value_1_1_0 = sample_array(array, spacing, origin, wp.vec3(position[0], position[1], position[2] - spacing[2]/2.0))
    value_2_1_1 = sample_array(array, spacing, origin, wp.vec3(position[0] + spacing[0]/2.0, position[1], position[2]))
    value_1_2_1 = sample_array(array, spacing, origin, wp.vec3(position[0], position[1] + spacing[1]/2.0, position[2]))
    value_1_1_2 = sample_array(array, spacing, origin, wp.vec3(position[0], position[1], position[2] + spacing[2]/2.0))

    # Compute the derivative
    array_dx = (value_2_1_1 - value_0_1_1) / spacing[0]
    array_dy = (value_1_2_1 - value_1_0_1) / spacing[1]
    array_dz = (value_1_1_2 - value_1_1_0) / spacing[2]

    # Return the derivative
    return wp.vec3(array_dx, array_dy, array_dz)


@wp.func
def ray_intersect_box(
        box_origin: wp.vec3,
        box_upper: wp.vec3,
        ray_origin: wp.vec3,
        ray_direction: wp.vec3):
    """Compute the intersection of a ray with a box.

    Parameters
    ----------
    box_origin : tuple
        The origin of the box
    box_upper : tuple
        The upper bounds of the box.
    ray_origin : tuple
        The origin of the ray.
    ray_direction : tuple
        The direction of the ray.
    """

    # Get tmix and tmax
    tmin_x = (box_origin[0] - ray_origin[0]) / ray_direction[0]
    tmax_x = (box_upper[0] - ray_origin[0]) / ray_direction[0]
    tmin_y = (box_origin[1] - ray_origin[1]) / ray_direction[1]
    tmax_y = (box_upper[1] - ray_origin[1]) / ray_direction[1]
    tmin_z = (box_origin[2] - ray_origin[2]) / ray_direction[2]
    tmax_z = (box_upper[2] - ray_origin[2]) / ray_direction[2]

    # Get tmin and tmax
    tmmin_x = wp.min(tmin_x, tmax_x)
    tmmax_x = wp.max(tmin_x, tmax_x)
    tmmin_y = wp.min(tmin_y, tmax_y)
    tmmax_y = wp.max(tmin_y, tmax_y)
    tmmin_z = wp.min(tmin_z, tmax_z)
    tmmax_z = wp.max(tmin_z, tmax_z)

    # Get t0 and t1
    t0 = wp.max(0.0, wp.max(tmmin_x, wp.max(tmmin_y, tmmin_z)))
    t1 = wp.min(tmmax_x, wp.min(tmmax_y, tmmax_z))

    # Return the intersection
    return t0, t1
