from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from PIL import Image
from shapely import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    contains_xy,
)

__all__ = [
    "T",
    "geometry_to_image",
    "get_image_dimensions",
    "get_shift_offsets",
    "plot_geometry",
    "single_to_tuple",
]

T = TypeVar("T")


def single_to_tuple[T](value: T | Sequence[T], typ: type[T], length: int) -> tuple[T, ...]:
    if isinstance(value, typ):
        return (value,) * length
    seq = cast("Sequence[T]", value)
    if not seq:
        raise ValueError("Cannot pad an empty sequence!")
    if len(seq) < length:
        return (*seq, *(seq[-1],) * (length - len(seq)))
    return tuple(seq)[:length]


if TYPE_CHECKING:
    from .extent import Extent
    from .shape import Shape


def get_image_dimensions(image: Image.Image, resolution: float) -> tuple[float, float]:
    shape = np.asarray(image).shape
    return (shape[1] * resolution, shape[0] * resolution)


def geometry_to_image(
    geom: Polygon | MultiPolygon,
    resolution: float = 1.0,
    extent: Extent | None = None,
    padding: float = 0.0,
    positive: bool = True,
) -> Image.Image:
    """Convert geometry to binary image using vectorized sampling."""

    if extent is None:
        extent = Extent(*geom.bounds).buffer(padding)

    if extent.get_sizes()[0] < resolution or extent.get_sizes()[1] < resolution:
        raise ValueError("ERROR: No geometry detected: likely rounding too high.")

    x = np.arange((extent[0] + 0.5 * resolution), extent[2] + 0.5 * resolution, resolution)  # type: ignore
    y = np.arange(extent[1] + 0.5 * resolution, extent[3] + 0.5 * resolution, resolution)  # type: ignore

    xx, yy = np.meshgrid(x, y)

    # Vectorized contains check
    mask = contains_xy(geom, xx.ravel(), yy.ravel())
    if positive:
        mask = 1 - mask
    image_array = mask.reshape(len(y), len(x)).astype(np.uint8) * 255

    # Flip vertically (Pillow uses top-left origin)
    image_array = np.flipud(image_array)

    # Create Monochrome PIL Image
    return Image.fromarray(image_array, mode="L").convert("1")


def get_shift_offsets(
    shift: float, units: Sequence[Shape] | Shape
) -> tuple[np.ndarray, np.ndarray]:
    # Angle offset
    try:
        units[0]  # type: ignore
    except TypeError:
        units = [units]  # type: ignore
    widths = np.array([u.get_extent().get_width() for u in units])  # type: ignore
    heights = np.array([u.get_extent().get_height() for u in units])  # type: ignore
    angle_offsets = np.rad2deg(np.arctan(-heights * shift / widths))

    # Period offsets
    period_offsets = np.sqrt(widths**2 + heights**2) - widths
    return period_offsets, angle_offsets


def plot_geometry(
    geom: Polygon | MultiPolygon | Shape,
    ax: Axes | None = None,
    fill_alpha: float = 0.3,
    fill_color: Any = None,
    edge_alpha: float = 1.0,
    edge_color: Any = None,
    marker: Any = None,
):
    """Plot any Shapely geometry."""

    if not isinstance(geom, Polygon | MultiPolygon):
        geom = geom.to_shapely()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if isinstance(geom, Point):
        ax.scatter(geom.x, geom.y, c=edge_color, alpha=edge_alpha, marker=marker)
    elif isinstance(geom, LineString):
        x, y = geom.xy
        ax.plot(x, y, color=edge_color, alpha=edge_alpha, marker=marker)
    elif isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.plot(x, y, color=edge_color, alpha=edge_alpha, marker=marker)
        ax.fill(x, y, alpha=fill_alpha, color=fill_color)
        # Plot holes if they exist
        for interior in geom.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=edge_color, alpha=edge_alpha, marker=marker)
            ax.fill(x, y, color="white")
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            plot_geometry(
                poly,
                ax,
                fill_alpha=fill_alpha,
                fill_color=fill_color,
                edge_alpha=edge_alpha,
                edge_color=edge_color,
                marker=marker,
            )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax
