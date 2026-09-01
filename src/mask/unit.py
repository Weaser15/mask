import numpy as np
from shapely import MultiPolygon, Point, Polygon, box
from shapely.affinity import rotate, scale, skew, translate
from shapely.ops import unary_union

from .shape import Shape


def ellipse(width: float, height: float, quad_segs: int = 16):
    # Start with a point, then buffer to create a unit circle and finally
    # scale to the correct shape.
    circle = Point(0.0, 0.0).buffer(0.5, quad_segs=quad_segs)
    ellipse = scale(circle, xfact=width, yfact=height)
    return Shape(ellipse)


def rectangle(width: float, height: float, rx: float = 0.0, ry: float = 0.0, quad_segs: int = 16):
    # If no rounded corners, it's handled simply by a box.
    if rx == 0.0 and ry == 0.0:
        sharp_rectangle = box(-0.5 * width, -0.5 * height, 0.5 * width, 0.5 * height)
        return Shape(sharp_rectangle)

    # If only one radius is specified, round using that one.
    if rx == 0.0:
        rx = ry
    elif ry == 0.0:
        ry = rx

    # Make sure that rx and ry cannot be larger than half of width and height respectively.
    rx = min(rx, width / 2)
    ry = min(ry, height / 2)

    # Want to create a rounded rectangle with corners (rx, ry).
    # The idea is that by dividing by (rx, ry), buffering by 1 and multiplying by (rx, ry),
    # the corners will be rounded correctly. Need to first subtract the amount gained.
    rd_width, rd_height = width - 2 * rx, height - 2 * ry
    sharp_rectangle = box(-0.5 * rd_width, -0.5 * rd_height, 0.5 * rd_width, 0.5 * rd_height)
    rd_rectangle = scale(sharp_rectangle, 1 / rx, 1 / ry)
    rnd_rd_rectangle = rd_rectangle.buffer(1.0, quad_segs=quad_segs)
    rectangle = scale(rnd_rd_rectangle, rx, ry)
    return Shape(rectangle)


def parallelogram(width: float, height: float, offset: float, angle: float | None = None):
    # Use angle if provided, otherwise compute the angle from the offset.
    ys = np.rad2deg(np.arctan(offset / width)) if angle is None else angle
    # The parallelogram is created by making a smaller rectangle then skewing.
    r_height = height - offset
    parallelogram = box(-0.5 * width, -0.5 * r_height, 0.5 * width, 0.5 * r_height)
    return Shape(skew(parallelogram, ys=ys))


def regular_polygon(radius: float, nsides: int):
    # Create a rectangular polygon by sampling a circle nsides number of times.
    angles = np.linspace(0, 2 * np.pi, nsides + 1)
    x, y = radius * np.sin(angles), radius * np.cos(angles)
    polygon = Polygon(zip(x, y))
    return Shape(polygon)
