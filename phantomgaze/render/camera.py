# Render functions for volumes

import warp as wp

#from phantomgaze.utils.math import normalize, dot, cross

@wp.func
def calculate_ray_direction(
        x: int,
        y: int,
        img_width: int,
        img_height: int,
        camera_position: wp.vec3,
        camera_focal: wp.vec3,
        camera_up: wp.vec3):
    """
    Calculate the direction of a ray from the camera to the image plane.

    Parameters
    ----------
    x : int
        The x coordinate of the pixel.
    y : int
        The y coordinate of the pixel.
    img_width : int
        The width of the image.
    img_height : int
        The height of the image.
    camera_position : tuple
        The position of the camera.
    camera_focal : tuple
        The focal point of the camera.
    camera_up : tuple
        The up vector of the camera.

    Returns
    -------
    ray_direction : tuple
    """

    # Compute base vectors
    forward = wp.vec3(
        camera_focal[0] - camera_position[0],
        camera_focal[1] - camera_position[1],
        camera_focal[2] - camera_position[2])
    
    forward = wp.normalize(forward)
    right = wp.cross(forward, camera_up)
    right = wp.normalize(right)
    up = wp.cross(right, forward)

    # Determine the center of the image
    center = wp.vec3(
        camera_position[0] + forward[0],
        camera_position[1] + forward[1],
        camera_position[2] + forward[2])

    # Calculate the location on the image plane corresponding (x, y)
    aspect_ratio = float(img_height) / float(img_width)
    s = (float(x) - float(img_height) / 2.0) / float(img_height)
    t = (float(y) - float(img_width) / 2.0) / float(img_width)

    # Adjust for aspect ratio and field of view (assuming 90 degrees here)
    s *= aspect_ratio * wp.tan(wp.pi / 4.0)
    t *= wp.tan(wp.pi / 4.0)
    point_on_image_plane = wp.vec3(
        center[0] + s * right[0] + t * up[0],
        center[1] + s * right[1] + t * up[1],
        center[2] + s * right[2] + t * up[2])

    # Calculate the ray direction
    ray_direction = wp.vec3(
        point_on_image_plane[0] - camera_position[0],
        point_on_image_plane[1] - camera_position[1],
        point_on_image_plane[2] - camera_position[2])
    
    ray_direction = wp.normalize(ray_direction)

    return ray_direction
