from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from shapely import MultiPolygon, Polygon
from shapely.affinity import rotate, scale, translate

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

    def union(
        self, other: Shape, extent: Literal["merge", "self", "other", "new"] = "merge"
    ) -> Shape:
        new_geom = self.to_shapely().union(other.to_shapely())
        assert isinstance(new_geom, Polygon | MultiPolygon)
        if extent.lower() == "merge":
            new_extent = self.get_extent().merge(other.get_extent())
        elif extent.lower() == "new":
            new_extent = Extent(*new_geom.bounds)
        elif extent.lower() == "self":
            new_extent = self.get_extent()
        elif extent.lower() == "other":
            new_extent = other.get_extent()
        else:
            message = f"extent must be one of (merge, new, self, other) not {extent}"
            raise KeyError(message)
        return Shape(new_geom, new_extent)

    def difference(self, other: Shape, extent: Literal["merge", "self", "other", "new"] = "new"):
        new_geom = self.to_shapely().difference(other.to_shapely())
        assert isinstance(new_geom, Polygon | MultiPolygon)
        if extent.lower() == "merge":
            new_extent = self.get_extent().merge(other.get_extent())
        elif extent.lower() == "new":
            new_extent = Extent(*new_geom.bounds)
        elif extent.lower() == "self":
            new_extent = self.get_extent()
        elif extent.lower() == "other":
            new_extent = other.get_extent()
        else:
            message = f"extent must be one of (merge, new, self, other) not {extent}"
            raise KeyError(message)
        return Shape(new_geom, new_extent)

    def scale(
        self,
        other: float | np.ndarray | list | tuple,
        origin: Literal["center"] | tuple[float, float] = "center",
    ):
        if isinstance(other, float):
            x_mul = other
            y_mul = other
        elif isinstance(other, (np.ndarray, list, tuple)):
            x_mul = other[0]
            y_mul = other[1]
        else:
            raise TypeError(f"Cannot divide by {type(other)}")
        new_extent = self.get_extent().scale(other, origin)

        new_geom = scale(self.to_shapely(), x_mul, y_mul, origin)  # type: ignore
        return Shape(new_geom, new_extent)

    def translate(
        self,
        x: float,
        y: float,
    ):
        new_extent = self.get_extent().translate(x, y)
        new_geom = translate(self.to_shapely(), x, y)
        return Shape(new_geom, new_extent)

    def rotate(self, angle: float, origin: Literal["center"] | tuple[float, float] = "center"):
        new_geom = rotate(self.to_shapely(), angle, origin=origin)
        new_extent = Extent(*new_geom.bounds)
        return Shape(new_geom, new_extent)
