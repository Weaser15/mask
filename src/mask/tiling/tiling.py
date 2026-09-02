from collections.abc import Iterable

import numpy as np

from ..extent import Extent


def _fill_periods(a1: float, a2: float | None, a3: float | None):
    return [a1, a1 if a2 is None else a2, a1 if a3 is None else a3]


class Tiling:
    def __init__(
        self,
        vectors: Iterable[Iterable[float]],
        coefficients: Iterable[Iterable[float]],
        rotations: Iterable[float],
        size_factors: Iterable[Iterable[float]],
    ):
        self._vectors = vectors
        self._coefficients = coefficients
        self._rotations = rotations
        self._size_factors = size_factors

    def get_positions(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        positions = sum(
            [
                period * np.asarray(vector) * np.asarray(positions)[:, np.newaxis]
                for (period, vector, positions) in zip(periods, self._vectors, self._coefficients)
            ]
        )
        assert isinstance(positions, np.ndarray)
        return positions

    def get_rotations(self, a1: float, a2: float | None = None, a3: float | None = None):
        return self._rotations

    def get_extent(self, a1: float, a2: float | None = None, a3: float | None = None):
        return Extent.from_sizes(self._get_sizes(a1, a2, a3))

    def _get_sizes(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        sizes = sum(
            [period * np.asarray(factor) for (period, factor) in zip(periods, self._size_factors)]
        )
        assert isinstance(sizes, np.ndarray)
        return sizes[0], sizes[1]


# Use angles between [-90, 0]. This way -90 (horizontal) will always be first
# and those closest to 0 (horizontal) will be last.

_vectors = [[1.0, 0.0], [0.0, 1.0]]
_coefficients = [[0.0], [0.0]]
_rotations = [0.0]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
rectangular = Tiling(_vectors, _coefficients, _rotations, _size_factors)


_vectors = [[1.0, 0.0], [0.0, 1.0]]
_coefficients = [[0.5, 0.0], [0.0, 0.5]]
_rotations = [-90.0, 0.0]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
square_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors)
