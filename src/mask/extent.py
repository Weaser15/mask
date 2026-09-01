"""Module containing Extent. It is responsible for storing where a shape is located."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np


class Extent(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    def merge(self, other: Extent):
        x0 = min(self.x0, other.x0)
        y0 = min(self.y0, other.y0)
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        return Extent(x0, y0, x1, y1)

    def scale(
        self,
        other: float | np.ndarray | list | tuple,
        origin: Literal["center"] | tuple[float, float] = "center",
    ) -> Extent:
        origin = self.get_centre() if origin == "center" else origin
        if isinstance(other, float):
            x_mul = other
            y_mul = other
        elif isinstance(other, (np.ndarray, list, tuple)):
            x_mul = other[0]
            y_mul = other[1]
        else:
            raise TypeError(f"Cannot divide by {type(other)}")
        new_x0 = (self.x0 - origin[0]) * x_mul
        new_y0 = (self.y0 - origin[1]) * y_mul
        new_x1 = (self.x1 - origin[0]) * x_mul
        new_y1 = (self.y1 - origin[1]) * y_mul

        return Extent(new_x0, new_y0, new_x1, new_y1)

    def get_width(self) -> float:
        """Return the width of the extent."""
        return self.x1 - self.x0

    def get_height(self) -> float:
        """Return the width of the extent."""
        return self.y1 - self.y0

    def get_x(self) -> tuple[float, float]:
        """Return the x coordinates of the extent."""
        return (self.x0, self.x1)

    def get_y(self) -> tuple[float, float]:
        """Return the y coordinates of the extent."""
        return (self.y0, self.y1)

    def get_p0(self) -> tuple[float, float]:
        """Return the first point of the extent."""
        return (self.x0, self.y0)

    def get_p1(self) -> tuple[float, float]:
        """Return the second point of the extent."""
        return (self.x1, self.y1)

    def get_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return both points of the extent."""
        return (self.get_p0(), self.get_p1())

    def get_centre(self) -> tuple[float, float]:
        """Return the centre point of the extent."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def get_sizes(self) -> tuple[float, float]:
        """Return the sizes (width, height) of the extent."""
        return (self.get_width(), self.get_height())

    def buffer(self, distance: float) -> Extent:
        """Add a distance to each side of the extent."""
        return Extent(
            self.x0 - distance, self.y0 - distance, self.x1 + distance, self.y1 + distance
        )

    def translate(self, x: float, y: float) -> Extent:
        """Translate the extent by (x, y)."""
        return Extent(self.x0 + x, self.y0 + y, self.x1 + x, self.y1 + y)

    def centre(self, x: float = 0.0, y: float = 0.0) -> Extent:
        """Centre the extent on (x, y). Centres on origin by default."""
        centre = self.get_centre()
        offset_x, offset_y = (x - centre[0], y - centre[1])
        return self.translate(offset_x, offset_y)

    def round(self, decimals: int = 0) -> Extent:
        """Round the points of the origin to the nearest decimal"""
        return Extent(*map(round, self, [decimals] * 4))

    def contains(self, other: Extent) -> bool:
        """Check whether this Extent fully contains another."""
        x_out = any(x <= self.x0 for x in other.get_x()) | any(x >= self.x1 for x in other.get_x())
        y_out = any(y <= self.y0 for y in other.get_y()) | any(y >= self.y1 for y in other.get_y())
        return not (x_out | y_out)

    @classmethod
    def from_sizes(cls, sizes: tuple[float, float]) -> Extent:
        """Build an extent centred around (0.0, 0.0) from the sizes (width, height)."""
        x0, x1 = -sizes[0] / 2, sizes[0] / 2
        y0, y1 = -sizes[1] / 2, sizes[1] / 2
        return cls(x0, y0, x1, y1)
