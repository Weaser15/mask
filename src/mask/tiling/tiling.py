from collections.abc import Iterable
from pathlib import Path

import numpy as np


class Tiling:
    def __init__(self, positions: Iterable[np.ndarray], rotations: Iterable[float], extent):
        self._positions = positions
        self._rotations = rotations
        self._extent = extent

    def get_positions(self, a1: float, a2: float | None = None, a3: float | None = None): ...

    def get_rotations(self, a1: float, a2: float | None = None, a3: float | None = None):
        return self._rotations

    def get_extent(self, a1: float, a2: float | None = None, a3: float | None = None): ...
