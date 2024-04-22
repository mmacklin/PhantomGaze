# Render functions for rendering a contour of a volume.

import cupy as cp
import warp as wp

from phantomgaze import ScreenBuffer
from phantomgaze import Colormap, SolidColor
from phantomgaze.render.camera import calculate_ray_direction
from phantomgaze.render.utils import sample_array, sample_array_derivative, ray_intersect_box
from phantomgaze.render.color import scalar_to_color

@wp.kernel
def contour_kernel(
        volume_array: wp.array3d(dtype=float),
        spacing: wp.vec3,
        origin: wp.vec3,
        camera_position: wp.vec3,
        camera_focal: wp.vec3,
        camera_up: wp.vec3,
        max_depth: float,
        threshold: float,
        color_array: wp.array3d(dtype=float),
        color_map_array: wp.array2d(dtype=float),
        vmin: float,
        vmax: float,
        nan_color: wp.vec3,
        nan_opacity: float,
        opaque: bool,
        opaque_pixel_buffer: wp.array3d(dtype=float),
        depth_buffer: wp.array2d(dtype=float),
        normal_buffer: wp.array3d(dtype=float),
        transparent_pixel_buffer: wp.array3d(dtype=float),
        revealage_buffer: wp.array2d(dtype=float)):

    """Kernel for rendering a contour of a volume.

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
        The maximum depth, used for Weighted Blended Order-Independent Transparency.
    threshold : float
        The threshold to use for the contour.
    color_array : ndarray
        The color data.
    color_map_array : ndarray
        The color map array.
    vmin : float
        The minimum value of the scalar range.
    vmax : float
        The maximum value of the scalar range.
    nan_color : tuple
        The color to use for NaN values.
    nan_opacity : float
        The opacity to use for NaN values.
    opaque : bool
        Whether the geometry is opaque or not.
    opaque_pixel_buffer : ndarray
        The opaque pixel buffer.
    depth_buffer : ndarray
        The depth buffer.
    normal_buffer : ndarray
        The normal buffer.
    transparent_pixel_buffer : ndarray
        The transparent pixel buffer.
    revealage_buffer : ndarray
        The reveal buffer.
    """

    # Get the x and y indices
    x, y = wp.tid()

    # Make sure the indices are in bounds
    if x >= opaque_pixel_buffer.shape[1] or y >= opaque_pixel_buffer.shape[0]:
        return

    # Get ray direction
    ray_direction = calculate_ray_direction(
            x, y, opaque_pixel_buffer.shape[0], opaque_pixel_buffer.shape[1],
            camera_position, camera_focal, camera_up)

    # Get volume upper bound
    volume_upper = wp.vec3(
        origin[0] + spacing[0] * float(volume_array.shape[0]),
        origin[1] + spacing[1] * float(volume_array.shape[1]),
        origin[2] + spacing[2] * float(volume_array.shape[2])
    )

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
        camera_position[2] + t0 * ray_direction[2]
    )

    # Get the step size
    step_size = wp.min(spacing[0], wp.min(spacing[1], spacing[2]))

    # Set starting value to lowest possible value
    value = sample_array(volume_array, spacing, origin, ray_pos)

    # Inside-outside stored in the sign
    sign = -1
    if value > threshold:
        sign = 1

    # Start the ray marching
    distance = t0
    for step in range(int((t1 - t0) / step_size)):
        # Check if distance is greater then current depth
        if (distance > depth_buffer[y, x]):
            return

        # Get next step position
        next_ray_pos = wp.vec3(
            ray_pos[0] + step_size * ray_direction[0],
            ray_pos[1] + step_size * ray_direction[1],
            ray_pos[2] + step_size * ray_direction[2]
        )

        # Get the value in the next step
        next_value = sample_array(volume_array, spacing, origin, next_ray_pos)

        # If contour is crossed, set the color and depth
        if (next_value - threshold) * float(sign) < 0.0:
            # Update the sign
            sign = -sign

            # Linearly interpolate the position
            t = (threshold - value) / (next_value - value)
            pos_contour = wp.vec3(
                ray_pos[0] + t * step_size * ray_direction[0],
                ray_pos[1] + t * step_size * ray_direction[1],
                ray_pos[2] + t * step_size * ray_direction[2]
            )

            # Get gradient
            gradient = sample_array_derivative(volume_array, spacing, origin, pos_contour)
            gradient = wp.normalize(gradient)

            # Calculate intensity
            intensity = wp.dot(gradient, ray_direction)
            intensity = wp.abs(intensity)

            # Get the color
            scalar = sample_array(color_array, spacing, origin, pos_contour)
            color = scalar_to_color(
                scalar, color_map_array, vmin, vmax)

            # if the color is nan, set it to the nan color
            if color[0] != color[0]:
                color = wp.vec4(nan_color[0], nan_color[1], nan_color[2], nan_opacity)

            # If opaque, set the opaque pixel buffer
            if opaque:
                # Set the opaque pixel buffer
                opaque_pixel_buffer[y, x, 0] = color[0] * intensity
                opaque_pixel_buffer[y, x, 1] = color[1] * intensity
                opaque_pixel_buffer[y, x, 2] = color[2] * intensity

                # Set the depth buffer
                depth_buffer[y, x] = distance

                # Set the normal buffer
                normal_buffer[y, x, 0] = gradient[0]
                normal_buffer[y, x, 1] = gradient[1]
                normal_buffer[y, x, 2] = gradient[2]

                # Exit the loop
                return

            # Else, use Weighted Blended Order-Independent Transparency
            else:
                # Calculate weight following Weighted Blended OIT
                normalized_distance = distance / max_depth
                weight = 1.0 / (normalized_distance**2.0 + 1.0)

                # Accumulate the color
                transparent_pixel_buffer[y, x, 0] += color[0] * weight * color[3] * intensity
                transparent_pixel_buffer[y, x, 1] += color[1] * weight * color[3] * intensity
                transparent_pixel_buffer[y, x, 2] += color[2] * weight * color[3] * intensity

                # Set the revealage buffer
                revealage_buffer[y, x] *= (1.0 - color[3] * weight)

        # Update the value and position
        value = next_value
        ray_pos = next_ray_pos
        distance += step_size


def contour(volume, camera, threshold, color=None, colormap=None, screen_buffer=None):
    """Render a contour of a volume.

    Parameters
    ----------
    volume : Volume
        The volume to render.
    camera : Camera
        The camera to render from.
    threshold : float
        The threshold to use for the contour.
    color : Volume, optional
        The color of the contour. If None, the color will be grey/white.
    colormap : Colormap, optional
        The colormap to use for the volume. If None, the colormap will be
        jet.
    screen_buffer : ScreenBuffer, optional
        The screen buffer to render to. If None, a new buffer will be created.
    """

    # Get the screen buffer
    if screen_buffer is None:
        screen_buffer = ScreenBuffer.from_camera(camera)

    # Get color data if necessary
    if color is None:
        color_array = cp.zeros(volume.shape, dtype=cp.float32)
        if colormap is None:
            colormap = SolidColor(color=(1.0, 1.0, 1.0, 1.0))
    else:
        color_array = color.array
        if colormap is None:

            host_color_array = color_array.numpy()

            range_min = host_color_array.min()
            range_max = host_color_array.max()

            colormap = Colormap('jet', float(range_min), float(range_max))

    # Run kernel
    wp.launch(contour_kernel, dim=[screen_buffer.width, screen_buffer.height], inputs=[
        volume.array,
        volume.spacing,
        volume.origin,
        camera.position,
        camera.focal_point,
        camera.view_up,
        camera.max_depth,
        threshold,
        color_array,
        colormap.color_map_array,
        colormap.vmin,
        colormap.vmax,
        colormap.nan_color,
        colormap.nan_opacity,
        colormap.opaque,
        screen_buffer.opaque_pixel_buffer,
        screen_buffer.depth_buffer,
        screen_buffer.normal_buffer,
        screen_buffer.transparent_pixel_buffer,
        screen_buffer.revealage_buffer]
    )

    return screen_buffer
