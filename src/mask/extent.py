from typing import NamedTuple


class Extent(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    def get_width(self):
        return self.x1 - self.x0

    def get_height(self):
        return self.y1 - self.y0

    def get_x(self):
        return (self.x0, self.x1)

    def get_y(self):
        return (self.y0, self.y1)

    def get_centre(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def get_dims(self):
        return (self.get_width(), self.get_height())

    def buffer(self, distance: float):
        return Extent(
            self.x0 - distance, self.y0 - distance, self.x1 + distance, self.y1 + distance
        )

    def offset(self, x: float, y: float):
        return Extent(self.x0 + x, self.y0 + y, self.x1 + x, self.y1 + y)

    def centre(self, x: float, y: float):
        centre = self.get_centre()
        offset_x, offset_y = (x - centre[0], y - centre[1])
        return self.offset(offset_x, offset_y)
