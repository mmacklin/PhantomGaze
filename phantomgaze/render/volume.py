# Render functions for volumes 

import cupy as cp
import numba
from numba import cuda

from phantomgaze import ScreenBuffer
from phantomgaze import Colormap, SolidColor
from phantomgaze.render.camera import calculate_ray_direction
from phantomgaze.render.utils import sample_array, sample_array_derivative, ray_intersect_box
from phantomgaze.render.color import scalar_to_color

import warp as wp

@wp.kernel
def volume_kernel(
        volume_array: wp.array3d(dtype=wp.float32),
        spacing: wp.vec3,
        origin: wp.vec3,
        camera_position: wp.vec3,
        camera_focal: wp.vec3,
        camera_up: wp.vec3,
        max_depth: float,
        color_map_array: wp.array2d(dtype=float),
        vmin: float,
        vmax: float,
        nan_color: wp.vec3,
        nan_opacity: float,
        depth_buffer: wp.array2d(dtype=float),
        transparent_pixel_buffer: wp.array3d(dtype=float),
        revealage_buffer: wp.array2d(dtype=float)):
    """Kernel for rendering a volume.

    Parameters
    ----------
    volume_array : ndarray
        The volume data.
    spacing : tuple
        The spacing of the volume data.
    origin : tuple
        The origin of the volume data.
    camera_position : tuple
        The position of the camera.
    camera_focal : tuple
        The focal point of the camera.
    camera_up : tuple
        The up vector of the camera.
    max_depth : float
        The maximum depth to render to.
    color_map_array : ndarray
        The color map data.
    vmin : float
        The minimum value of the volume.
    vmax : float
        The maximum value of the volume.
    nan_color : tuple
        The color to use for NaN values.
    nan_opacity : float
        The opacity to use for NaN values.
    depth_buffer : ndarray
        The buffer to store depth values in.
    transparent_pixel_buffer : ndarray
        The buffer to store transparent pixels in.
    revealage_buffer : ndarray
        The buffer to store revealage values in.
    """

    # Get the x and y indices
    x, y = wp.tid()

    # Make sure the indices are in bounds
    if x >= transparent_pixel_buffer.shape[1] or y >= transparent_pixel_buffer.shape[0]:
        return

    # Get ray direction
    ray_direction = calculate_ray_direction(
            x, y, transparent_pixel_buffer.shape[0], transparent_pixel_buffer.shape[1],
            camera_position, camera_focal, camera_up)

    # Get volume upper bound
    volume_upper = wp.vec3(
        origin[0] + spacing[0] * float(volume_array.shape[0]),
        origin[1] + spacing[1] * float(volume_array.shape[1]),
        origin[2] + spacing[2] * float(volume_array.shape[2]))

    # Get the intersection of the ray with the volume
    t0, t1 = ray_intersect_box(
        origin, volume_upper, camera_position, ray_direction)

    # If there is no intersection, return
    if t0 > t1:
        return

    # Get the starting point of the ray
    ray_pos = wp.vec3(
        camera_position[0] + t0 * ray_direction[0],
        camera_position[1] + t0 * ray_direction[1],
        camera_position[2] + t0 * ray_direction[2])

    # Get the step size
    step_size = wp.min(spacing[0], wp.min(spacing[1], spacing[2]))

    # Start the ray marching
    distance = t0
    for step in range(int((t1 - t0) / step_size)):
        # Break out of the kernel if the distance is greater than the depth
        if distance > depth_buffer[y, x]:
            break

        # Get the value at the current position
        value = sample_array(volume_array, spacing, origin, ray_pos)

        # Get the color
        color = scalar_to_color(value, color_map_array, vmin, vmax)

        # Calculate weight following Weighted Blended OIT
        normalized_distance = distance / max_depth
        weight = 1.0 / (normalized_distance**2.0 + 1.0)

        # Accumulate the color
        transparent_pixel_buffer[y, x, 0] += (
            color[0] * weight * color[3] * step_size
        )
        transparent_pixel_buffer[y, x, 1] += (
            color[1] * weight * color[3] * step_size
        )
        transparent_pixel_buffer[y, x, 2] += (
            color[2] * weight * color[3] * step_size
        )

        # Accumulate the revealage
        revealage_buffer[y, x] *= (1.0 - color[3] * weight * step_size)

        # Increment the distance
        ray_pos = wp.vec3(
            ray_pos[0] + step_size * ray_direction[0],
            ray_pos[1] + step_size * ray_direction[1],
            ray_pos[2] + step_size * ray_direction[2])
        
        distance += step_size


def volume(volume, camera, colormap=None, screen_buffer=None):
    """Render a volume

    Parameters
    ----------
    volume : phantom.object.Volume
        The volume to render.
    camera : Camera
        The camera to render with.
    colormap : ColorMap
        The colormap to use for the volume. If None, the colormap will be
        jet with the minimum value of the volume as the minimum value and the
        maximum value of the volume as the maximum value.
    screen_buffer : ndarray
        The buffer to render to.
    """

    # Get the screen buffer
    if screen_buffer is None:
        screen_buffer = ScreenBuffer.from_camera(camera)

    # Get colormap if necessary
    if colormap is None:
        colormap = Colormap('jet', float(volume.array.min()), float(volume.array.max()))

    # Run kernel
    wp.launch(volume_kernel, dim=[screen_buffer.width, screen_buffer.height], inputs=[
        volume.array,
        volume.spacing,
        volume.origin,
        camera.position,
        camera.focal_point,
        camera.view_up,
        camera.max_depth,
        colormap.color_map_array,
        colormap.vmin,
        colormap.vmax,
        colormap.nan_color,
        colormap.nan_opacity,
        screen_buffer.depth_buffer,
        screen_buffer.transparent_pixel_buffer,
        screen_buffer.revealage_buffer]
    )

    return screen_buffer
