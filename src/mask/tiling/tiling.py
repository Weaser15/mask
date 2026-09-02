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
                period * np.asarray(vector) * np.asarray(coeff)[:, np.newaxis]
                for (period, vector, coeff) in zip(
                    periods, self._vectors, np.asarray(self._coefficients).T
                )
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

        axes_contributions = (
            np.asarray(self._size_factors) * np.asarray(self._vectors).T * np.asarray(periods)
        )
        sizes = axes_contributions.sum(axis=1)
        assert isinstance(sizes, np.ndarray)
        return sizes[0], sizes[1]


def _get_av(angle: float):
    """Produces vectors for an angle.
    Produces [1.0, 0.0] for 0 and [0.0, 1.0] for 90.
    """
    return np.round([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))], 5)


# Use angles between [0, 180]. This way 0 (horizontal) will always be first
# and those closest to 90 (vertical) will be last.

_a1, _a2 = 0, 90
_vectors = [_get_av(_a1), _get_av(_a2)]
_coefficients = [[0.0], [0.0]]
_rotations = [_a1]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
rectangular = Tiling(_vectors, _coefficients, _rotations, _size_factors)


_a1, _a2 = 0.0, 90.0
_vectors = [_get_av(_a1), _get_av(_a2)]
_coefficients = [[0.5, 0.0], [0.0, 0.5]]
_rotations = [_a1, _a2]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
square_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors)


_a1, _a2, _a3 = 0.0, 60.0, 120.0
_vectors = [_get_av(_a1), _get_av(_a2), _get_av(_a3)]
_coefficients = [
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.0, -0.5, 0.0],
    [0.0, 0.0, -0.5],
    [-0.5, 0.0, 1.0],
]
_rotations = [_a1, _a2, _a3, _a2, _a3, _a1]
_size_factors = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
trigonal_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors)

# _vectors = [[1.0, 0.0], [0.0, 1.0]]
# _coefficients = [[0.5, 0.0], [0.0, 0.5]]
# _rotations = [-90.0, 0.0]
# _size_factors = [[1.0, 0.0], [0.0, 1.0]]
# square_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors)
