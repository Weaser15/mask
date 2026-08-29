from collections.abc import Iterable

import numpy as np


def _fill_periods(a1: float, a2: float | None, a3: float | None):
    return [a1, a1 if a2 is None else a2, a1 if a3 is None else a3]

class Tiling:
    def __init__(self, vectors: Iterable[np.ndarray], positions: Iterable[np.ndarray], rotations: Iterable[float], extent: np.ndarray):
        self._vectors = vectors
        self._positions = positions
        self._rotations = rotations
        self._extent = extent

    def get_positions(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        return [period * vector * positions for (period, vector, positions) in zip(periods, self._vectors, self._positions)]


    def get_rotations(self, a1: float, a2: float | None = None, a3: float | None = None):
        return self._rotations

    def get_extent(self, a1: float, a2: float | None = None, a3: float | None = None): 
        periods = _fill_periods(a1, a2, a3)
        return [period * extent for (period, extent) in zip(periods, self._extent)]


