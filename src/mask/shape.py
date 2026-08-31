from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from shapely import MultiPolygon, Polygon
from shapely.affinity import scale

from .extent import Extent
from .utils import geometry_to_image, get_image_dimensions, plot_geometry


class Shape:
    def __init__(self, geom: Polygon | MultiPolygon, extent: Extent | None = None):
        self._geom = geom
        self._extent = Extent(*geom.bounds) if extent is None else extent

    def get_extent(self):
        return self._extent

    def to_shapely(self):
        return self._geom

    def plot(
        self,
        ax: Axes | None = None,
        fill_alpha: float = 0.3,
        fill_color: Any = "dodgerblue",
        edge_alpha: float = 1.0,
        edge_color: Any = "dodgerblue",
        marker: Any = None,
    ):

        ax = plot_geometry(
            geom=self.to_shapely(),
            ax=ax,
            fill_alpha=fill_alpha,
            fill_color=fill_color,
            edge_alpha=edge_alpha,
            edge_color=edge_color,
            marker=marker,
        )
        ax.set_xlim(*self.get_extent().get_x())  # type: ignore
        ax.set_ylim(*self.get_extent().get_y())  # type: ignore
        return ax

    def to_image(
        self,
        filepath: str | Path | None = None,
        resolution: float = 1.0,
        print_dimensions: bool = True,
    ):

        # Round everything to match the resolution
        extent = self.get_extent()
        new_extent = extent.scale(1 / resolution).round().scale(resolution)
        xfact = new_extent.get_width() / extent.get_width()
        yfact = new_extent.get_height() / extent.get_height()
        geom = scale(self.to_shapely(), xfact, yfact)
        image = geometry_to_image(geom, resolution, new_extent)
        if filepath is not None:
            image.save(filepath)
        if print_dimensions:
            print("{:g}nm x {:g}nm".format(*get_image_dimensions(image, resolution)))

        return image
