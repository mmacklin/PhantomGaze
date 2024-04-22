# Class that stores fragment information for rendering

import cupy as cp
import numba

import warp as wp

# Make kernel for combining buffers
@wp.kernel
def _combine_buffers_kernel(
        opaque_pixel_buffer: wp.array3d(dtype=float),
        depth_buffer: wp.array2d(dtype=float),
        normal_buffer: wp.array3d(dtype=float),
        transparent_pixel_buffer: wp.array3d(dtype=float),
        revealage_buffer: wp.array2d(dtype=float),
        background_buffer: wp.array3d(dtype=float),
        image_buffer: wp.array3d(dtype=float)):

    # Get the x and y indices
    x, y = wp.tid()

    # Make sure the indices are in bounds
    if x >= image_buffer.shape[1] or y >= image_buffer.shape[0]:
        return

    # If the depth is not infinite, then the pixel is opaque
    if depth_buffer[y, x] != wp.inf:
        # Get the color from the opaque pixel buffer
        final_color = wp.vec3(
            opaque_pixel_buffer[y, x, 0],
            opaque_pixel_buffer[y, x, 1],
            opaque_pixel_buffer[y, x, 2])
    else:
        # Initialize from background
        final_color = wp.vec3(
            background_buffer[y, x, 0],
            background_buffer[y, x, 1],
            background_buffer[y, x, 2])

    # Blend with revealage
    alpha = (1.0 - revealage_buffer[y, x])
    transparent_color = wp.vec3(
        transparent_pixel_buffer[y, x, 0],
        transparent_pixel_buffer[y, x, 1],
        transparent_pixel_buffer[y, x, 2])
    
    final_color = wp.vec3(
        final_color[0] * (1.0 - alpha) + transparent_color[0] * alpha,
        final_color[1] * (1.0 - alpha) + transparent_color[1] * alpha,
        final_color[2] * (1.0 - alpha) + transparent_color[2] * alpha)

    # Write to the image buffer (flip y axis TODO: fix this)
    flip_y = image_buffer.shape[0] - y - 1
    image_buffer[flip_y, x, 0] = final_color[0]
    image_buffer[flip_y, x, 1] = final_color[1]
    image_buffer[flip_y, x, 2] = final_color[2]
    image_buffer[flip_y, x, 3] = 1.0 


class ScreenBuffer:
    """
    Create a screen buffer.
    The screen buffer stores fragment information for rendering.

    Parameters
    ----------
    height : int
        The height of the screen buffer
    width : int
        The width of the screen buffer
    """

    def __init__(self, height, width):
        # Store height and width
        self.height = height
        self.width = width

        # Create buffers for opaque rendering
        self.opaque_pixel_buffer = wp.zeros((height, width, 3), dtype=wp.float32)
        self.depth_buffer = wp.full(value=wp.inf, shape=(height, width), dtype=wp.float32)
        self.normal_buffer = wp.zeros((height, width, 3), dtype=wp.float32)

        # Create buffer transparent rendering
        self.transparent_pixel_buffer = wp.zeros((height, width, 3), dtype=wp.float32)
        self.revealage_buffer = wp.ones((height, width), dtype=wp.float32)

        # Create buffer for background
        self.background_buffer = wp.zeros((height, width, 3), dtype=wp.float32)

        # Create buffer for final image
        self.image_buffer = wp.zeros((height, width, 4), dtype=wp.float32)

    @staticmethod
    def from_camera(camera):
        """ Create a screen buffer from a camera

        Parameters
        ----------
        camera : Camera
            The camera to create the screen buffer for, uses the camera's height and width
        """

        # Make screen buffer
        screen_buffer = ScreenBuffer(camera.height, camera.width)

        # Set background
        #screen_buffer.background_buffer.fill_(camera.background.color[0])
        screen_buffer.background_buffer[:, :, 0].fill_(camera.background.color[0])
        screen_buffer.background_buffer[:, :, 1].fill_(camera.background.color[1])
        screen_buffer.background_buffer[:, :, 2].fill_(camera.background.color[2])

        return screen_buffer

    @property
    def image(self):
        """ Get the image buffer """

        wp.launch(_combine_buffers_kernel, dim=[self.width, self.height], inputs=[
            self.opaque_pixel_buffer,
            self.depth_buffer,
            self.normal_buffer,
            self.transparent_pixel_buffer,
            self.revealage_buffer,
            self.background_buffer,
            self.image_buffer]
        )
        return self.image_buffer

    def clear(self):
        """ Clear the screen buffer """

        # Clear opaque buffers
        self.opaque_pixel_buffer.fill_(0.0)
        self.depth_buffer.fill_(wp.inf)
        self.normal_buffer.fill_(0.0)

        # Clear transparent buffers
        self.transparent_pixel_buffer.fill_(0.0)
        self.revealage_buffer.fill_(1.0)

        # Clear background buffer
        self.background_buffer.fill_(0.0)

        # Clear image buffer
        self.image_buffer.fill_(0.0)
