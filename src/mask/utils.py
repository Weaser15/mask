from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from .extent import Extent
    from .shape import Shape


def get_image_dimensions(image: Image.Image, resolution: float) -> tuple[float, float]:
    shape = np.asarray(image).shape
    return (shape[0] * resolution, shape[1] * resolution)


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
