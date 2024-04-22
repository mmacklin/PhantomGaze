# Geometries

import warp as wp
import math

class Geometry:
    """
    Class that represents a geometry to be rendered
    The geometry is defined by a signed distance function
    This allows for boolean operations to be performed on the geometry

    Parameters
    ----------
    sdf : Warp function
        The signed distance function
    lower_bound : tuple
        Lower bound of the signed distance function
    upper_bound : tuple
        Upper bound of the signed distance function
    distance_threshold : float
        The distance threshold for the signed distance function when rendering
    """

    def __init__(self, sdf, lower_bound, upper_bound, distance_threshold=0.001):
        # Store the parameters
        self.sdf = sdf
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.distance_threshold = distance_threshold

        # Define the kernel for the derivative        
        def sdf_derivative(pos: wp.vec3):
            dx = sdf(wp.vec3(pos[0] + 0.001, pos[1], pos[2])) - sdf(
                wp.vec3(pos[0] - 0.001, pos[1], pos[2])
            )
            dy = sdf(wp.vec3(pos[0], pos[1] + 0.001, pos[2])) - sdf(
                wp.vec3(pos[0], pos[1] - 0.001, pos[2])
            )
            dz = sdf(wp.vec3(pos[0], pos[1], pos[2] + 0.001)) - sdf(
                wp.vec3(pos[0], pos[1], pos[2] - 0.001)
            )
            return wp.vec3(dx, dy, dz)
        
        self.derivative = wp.Function(sdf_derivative, key=f"sdf_derivative_{id(self)}", namespace="")

    def __add__(self, other):
        """
        Adds two signed distance functions together (union)

        Parameters
        ----------
        other : Geometry 
            The other signed distance function

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Get the signed distance functions
        sdf = self.sdf
        other_sdf = other.sdf

        # Define the new signed distance function        
        def new_sdf(pos: wp.vec3):
            return wp.min(sdf(pos), other_sdf(pos))

        new_sdf = wp.Function(new_sdf, key=f"sdf_union_{id(self)}", namespace="")

        # Expand the bounds
        lower_bound = (
            min(self.lower_bound[0], other.lower_bound[0]),
            min(self.lower_bound[1], other.lower_bound[1]),
            min(self.lower_bound[2], other.lower_bound[2]),
        )
        upper_bound = (
            max(self.upper_bound[0], other.upper_bound[0]),
            max(self.upper_bound[1], other.upper_bound[1]),
            max(self.upper_bound[2], other.upper_bound[2]),
        )

        # Return the new signed distance function
        return Geometry(new_sdf, lower_bound, upper_bound, min(self.distance_threshold, other.distance_threshold))

    def __sub__(self, other):
        """
        Subtracts two signed distance functions together (difference)

        Parameters
        ----------
        other : Geometry
            The other signed distance function

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Define the new signed distance function
        def new_sdf(pos: wp.vec3):
            return wp.max(self.sdf(pos), -other.sdf(pos))

        new_sdf = wp.Function(new_sdf, key=f"sdf_difference_{id(self)}", namespace="")

        # Return the new signed distance function
        return Geometry(new_sdf, self.lower_bound, self.upper_bound, self.distance_threshold)

    def __and__(self, other):
        """
        Intersects two signed distance functions together (intersection)

        Parameters
        ----------
        other : Geometry
            The other signed distance function

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Define the new signed distance function
        def new_sdf(pos: wp.vec3):
            return wp.max(self.sdf(pos), other.sdf(pos))

        new_sdf = wp.Function(new_sdf, key=f"sdf_intersect_{id(self)}", namespace="")

        # Shrink the bounds
        lower_bound = (
            max(self.lower_bound[0], other.lower_bound[0]),
            max(self.lower_bound[1], other.lower_bound[1]),
            max(self.lower_bound[2], other.lower_bound[2]),
        )
        upper_bound = (
            min(self.upper_bound[0], other.upper_bound[0]),
            min(self.upper_bound[1], other.upper_bound[1]),
            min(self.upper_bound[2], other.upper_bound[2]),
        )

        # Return the new signed distance function
        return Geometry(new_sdf, lower_bound, upper_bound, min(self.distance_threshold, other.distance_threshold))

    def translate(self, translation):
        """
        Translates the signed distance function

        Parameters
        ----------
        translation : tuple
            The translation vector

        Returns
        -------
        Geometry
            The new signed distance function
        """

        # Get the signed distance function
        sdf = self.sdf

        translation = wp.constant(wp.vec3(translation))

        # Define the new signed distance function        
        def new_sdf(pos: wp.vec3):
            return sdf(wp.vec3(pos[0] - translation[0], pos[1] - translation[1], pos[2] - translation[2]))
        
        new_sdf = wp.Function(new_sdf, key=f"translate_sdf_{id(self)}", namespace="")

        # Get new bounds
        lower_bound = (
            self.lower_bound[0] + translation[0],
            self.lower_bound[1] + translation[1],
            self.lower_bound[2] + translation[2],
        )
        upper_bound = (
            self.upper_bound[0] + translation[0],
            self.upper_bound[1] + translation[1],
            self.upper_bound[2] + translation[2],
        )

        # Return the new signed distance function
        return Geometry(new_sdf, lower_bound, upper_bound, self.distance_threshold)


    def rotate(self, angle, axis):
        """
        Rotates the signed distance function

        Parameters
        ----------
        angle : float
            The angle to rotate by
        axis : tuple
            The axis to rotate around

        Returns
        -------
        Geometry
            The new signed distance function
        """


        # Compute the rotation quaternion
        half_angle = angle / 2.0
        c = math.cos(half_angle)
        s = math.sin(half_angle)
        q = (c, axis[0] * s, axis[1] * s, axis[2] * s) # rotation quaternion
        q_inv = (c, -axis[0] * s, -axis[1] * s, -axis[2] * s) # inverse of rotation quaternion

        # Get the signed distance function
        sdf = self.sdf

        # convert to Warp quaternion convention of i, j, k, r
        qx = wp.constant(q[1])
        qy = wp.constant(q[2])
        qz = wp.constant(q[3])
        qw = wp.constant(q[0])

        # Define the new signed distance function
        #@wp.func
        def new_sdf(pos: wp.vec3):
               
            # Perform quaternion multiplication: q * pos * q_inv
            q = wp.quat(qx, qy, qz, qw)
            pos_rotated = wp.quat_rotate(q, pos)
    
            # Extract the rotated position vector
            return sdf(pos_rotated)

        new_sdf = wp.Function(new_sdf, key=f"rotate_sdf_{id(self)}", namespace="")

        # Return the new signed distance function
        # TODO: Fix the bounds
        return Geometry(new_sdf, self.lower_bound, self.upper_bound, self.distance_threshold)


class Sphere(Geometry):
    """
    Class that represents a sphere

    Parameters
    ----------
    radius : float
        The radius of the sphere
    center : tuple
        The center of the sphere
    """

    def __init__(self, radius, center=(0.0, 0.0, 0.0)):
        
        center = wp.constant(wp.vec3(center))

        # Define the signed distance function
        def sdf(pos: wp.vec3):
            return wp.length(pos - center) - radius

        sdf = wp.Function(sdf, key=f"sdf_sphere_{id(self)}", namespace="")

        # Define the bounds
        lower_bound = (-radius+center[0], -radius+center[1], -radius+center[2])
        upper_bound = (radius+center[0], radius+center[1], radius+center[2])

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=radius / 100.0)


class BoxFrame(Geometry):
    """
    Class that represents a box frame
    Reference: https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    lower_bound : tuple
        lower bound of the box frame
    upper_bound : tuple
        upper bound of the box frame
    thickness : float
        thickness of the box frame
    """

    def __init__(self, lower_bound, upper_bound, thickness):
        

        lower = wp.constant(wp.vec3(lower_bound))
        upper = wp.constant(wp.vec3(upper_bound))
        thickness = wp.constant(thickness)

        # Define the signed distance function
        def sdf(pos: wp.vec3):
            # Compute size of the box frame
            size = wp.vec3(upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2])
        
            # Compute the center of the box frame
            center = wp.vec3(lower[0] + size[0] / 2.0, lower[1] + size[1] / 2.0, lower[2] + size[2] / 2.0)
        
            # Compute the point position relative to the center
            p = wp.vec3(pos[0] - center[0], pos[1] - center[1], pos[2] - center[2])
        
            # Compute the SDF
            p = wp.vec3(abs(p[0]) - size[0] / 2.0, abs(p[1]) - size[1] / 2.0, abs(p[2]) - size[2] / 2.0)
            q = wp.vec3(
                abs(p[0] + thickness) - thickness,
                abs(p[1] + thickness) - thickness,
                abs(p[2] + thickness) - thickness)
            
            e_x = wp.length(wp.vec3(wp.max(p[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))) + wp.min(
                wp.max(p[0], wp.max(q[1], q[2])), 0.0
            )
            e_y = wp.length(wp.vec3(wp.max(q[0], 0.0), wp.max(p[1], 0.0), wp.max(q[2], 0.0))) + wp.min(
                wp.max(q[0], wp.max(p[1], q[2])), 0.0
            )
            e_z = wp.length(wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(p[2], 0.0))) + wp.min(
                wp.max(q[0], wp.max(q[1], p[2])), 0.0
            )
        
            return wp.min(wp.min(e_x, e_y), e_z)

        sdf = wp.Function(sdf, key=f"sdf_box_frame_{id(self)}", namespace="")

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=thickness/100.0)

class Cone(Geometry):
    """
    Class that represents a cone
    Reference: https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    c: tuple
        The (sin, cos) of the angle
    h: float
        The height of the cone
    center: tuple
        The center of the cone
    """

    def __init__(self, c, h, center=(0.0, 0.0, 0.0)):
        
        c = wp.constant(wp.vec2(c))
        h = wp.constant(h)
        center = wp.constant(wp.vec3(center))

        # Define the signed distance function
        def sdf(pos: wp.vec3):
            # Compute the point position relative to the center
            p = wp.vec3(pos[0] - center[0], pos[1] - center[1], pos[2] - center[2])

            # Compute the SDF
            q = wp.vec2(h * c[0] / c[1], -h)
            w = wp.vec2(wp.length(wp.vec2(p[0], p[2])), p[1])
            a = wp.vec2(w[0] - q[0] * wp.clamp(wp.dot(w, q) / wp.dot(q, q), 0.0, 1.0), w[1] - q[1] * wp.clamp(wp.dot(w, q) / wp.dot(q, q), 0.0, 1.0))
            b = wp.vec2(w[0] - q[0] * wp.clamp(w[0] / q[0], 0.0, 1.0), w[1] - q[1] * 1.0)
            k = wp.sign(q[1])
            d = wp.min(wp.dot(a, a), wp.dot(b, b))
            s = wp.max(k * (w[0] * q[1] - w[1] * q[0]), k * (w[1] - q[1]))
            return d**0.5 * wp.sign(s)

        sdf = wp.Function(sdf, key=f"sdf_cone_{id(self)}", namespace="")

        # Define the bounds
        lower_bound = (-h + center[0], -h + center[1], -h + center[2])
        upper_bound = (h + center[0], h + center[1], h + center[2])

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=h / 100.0)

class Cylinder(Geometry):
    """
    Class that represents a cylinder
    Reference: https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    radius: float
        The radius of the cylinder
    height: float
        The height of the cylinder
    center: tuple
        The center of the cylinder
    """

    def __init__(self, radius, height, center=(0.0, 0.0, 0.0)):
        
        radius = wp.constant(radius)
        height = wp.constant(height)
        center = wp.constant(wp.vec3(center))

        # Define the signed distance function
        def sdf(pos: wp.vec3):
            # Compute the point position relative to the center
            p = pos - center

            # Compute the SDF
            d = wp.length(wp.vec2(p[0], p[2])) - radius
            h = wp.abs(p[1]) - height
            return wp.min(wp.max(d, h), 0.0) + wp.length(wp.vec2(wp.max(d, 0.0), wp.max(h, 0.0)))

        sdf = wp.Function(sdf, key=f"sdf_cylinder_{id(self)}", namespace="")

        # Define the bounds
        lower_bound = (-radius + center[0], -height + center[1], -radius + center[2])
        upper_bound = (radius + center[0], height + center[1], radius + center[2])

        # Call the parent constructor
        super().__init__(sdf, lower_bound, upper_bound, distance_threshold=radius / 100.0)

class Arrow(Geometry):
    """
    Class that represents an arrow

    Parameters
    ----------
    height: float
        The height of the arrow
    center: tuple
        The center of the arrow
    """

    def __init__(self, height, center=(0.0, 0.0, 0.0)):
        # Define the radius
        radius = height / 10.0

        # Make the cylinder
        cylinder = Cylinder(radius, height)

        # Make the cone
        angle = math.atan2(radius, height/3.0)
        cone = Cone((math.sin(angle), math.cos(angle)), height, (0.0 , 3.0 * height / 2.0, 0.0))
        cone = cone.rotate(math.pi, (1.0, 0.0, 0.0))

        # Union the two geometries
        arrow = cylinder + cone
        arrow = arrow.translate(center)

        # Call the parent constructor
        super().__init__(arrow.sdf, arrow.lower_bound, arrow.upper_bound, distance_threshold=radius / 100.0)

