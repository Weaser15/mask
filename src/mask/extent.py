from __future__ import annotations

from typing import NamedTuple


class Extent(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    def get_width(self) -> float:
        return self.x1 - self.x0

    def get_height(self) -> float:
        return self.y1 - self.y0

    def get_x(self) -> tuple[float, float]:
        return (self.x0, self.x1)

    def get_y(self) -> tuple[float, float]:
        return (self.y0, self.y1)

    def get_centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def get_sizes(self) -> tuple[float, float]:
        return (self.get_width(), self.get_height())

    def buffer(self, distance: float) -> Extent:
        return Extent(
            self.x0 - distance, self.y0 - distance, self.x1 + distance, self.y1 + distance
        )

    def offset(self, x: float, y: float) -> Extent:
        return Extent(self.x0 + x, self.y0 + y, self.x1 + x, self.y1 + y)

    def centre(self, x: float, y: float) -> Extent:
        centre = self.get_centre()
        offset_x, offset_y = (x - centre[0], y - centre[1])
        return self.offset(offset_x, offset_y)

    def round(self, decimals: int = 0) -> Extent:
        return Extent(*map(round, self, [decimals] * 4))

    def contains(self, other: Extent) -> bool:
        x_out = any(x <= self.x0 for x in other.get_x()) | any(x >= self.x1 for x in other.get_x())
        y_out = any(y <= self.y0 for y in other.get_y()) | any(y >= self.y1 for y in other.get_y())
        return not (x_out | y_out)

    @classmethod
    def from_sizes(cls, sizes: tuple[float, float]) -> Extent:
        x0, x1 = -sizes[0] / 2, sizes[0] / 2
        y0, y1 = -sizes[1] / 2, sizes[1] / 2
        return cls(x0, y0, x1, y1)
