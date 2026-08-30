from collections.abc import Iterable

import numpy as np


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
        return sum(
            [
                period * np.asarray(vector) * np.asarray(positions)[:, np.newaxis]
                for (period, vector, positions) in zip(periods, self._vectors, self._coefficients)
            ]
        )

    def get_rotations(self, a1: float, a2: float | None = None, a3: float | None = None):
        return self._rotations

    def get_extent(self, a1: float, a2: float | None = None, a3: float | None = None):
        size = self._get_size(a1, a2, a3)
        pass
        # return Extent.from_size(size)

    def _get_size(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        return sum(
            [period * np.asarray(factor) for (period, factor) in zip(periods, self._size_factors)]
        )


_vectors = [[1.0, 0.0], [0.0, 1.0]]
_coefficients = [[0.0], [0.0]]
_rotations = [0.0]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
rectangular = Tiling(_vectors, _coefficients, _rotations, _size_factors)


_vectors = [[1.0, 0.0], [0.0, 1.0]]
_coefficients = [[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, -1.0, 0.0]]
_rotations = [0.0, 90.0, 180.0, 270.0]
_size_factors = [[2.0, 0.0], [0.0, 2.0]]
square_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors)
