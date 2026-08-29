from collections.abc import Iterable

import numpy as np


def _fill_periods(a1: float, a2: float | None, a3: float | None):
    return [a1, a1 if a2 is None else a2, a1 if a3 is None else a3]

class Tiling:
    def __init__(self, vectors: Iterable[Iterable[float]], positions: Iterable[Iterable[float]], rotations: Iterable[float], extent: Iterable[Iterable[float]]):
        self._vectors = vectors
        self._positions = positions
        self._rotations = rotations
        self._extent = extent

    def get_positions(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        return sum([period * np.asarray(vector) * np.asarray(positions) for (period, vector, positions) in zip(periods, self._vectors, self._positions)])


    def get_rotations(self, a1: float, a2: float | None = None, a3: float | None = None):
        return self._rotations

    def get_extent(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        return sum([period * np.asarray(extent) for (period, extent) in zip(periods, self._extent)])

_vectors = [[0.0, 1.0], [1.0, 0.0]]
_positions = [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0 -1.0]]
_rotations = [0.0, 90.0, 180.0, 270.0]
_extent = [[2.0, 0.0], [0.0, 2.0]]
square_spin_ice = Tiling(_vectors, _positions, _rotations, _extent)