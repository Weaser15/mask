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
        n_periods: int,
        n_units: int,
    ):
        self._vectors = np.asarray(vectors)
        self._coefficients = np.asarray(coefficients)
        self._rotations = np.asarray(rotations)
        self._size_factors = np.asarray(size_factors)
        self._n_periods = n_periods
        self._n_units = n_units

    def get_positions(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)
        vectors = self.get_vectors()
        positions = sum(
            [
                period * vector * coeff[:, np.newaxis]
                for (period, vector, coeff) in zip(periods, vectors, self._coefficients.T)
            ]
        )
        assert isinstance(positions, np.ndarray)
        return positions.copy()

    def get_vectors(self):
        return self._vectors.copy()

    def get_rotations(self, a1: float, a2: float | None = None, a3: float | None = None):
        return self._rotations.copy()

    def get_extent(self, a1: float, a2: float | None = None, a3: float | None = None):
        return Extent.from_sizes(self._get_sizes(a1, a2, a3))

    def get_indices(self, a1: float, a2: float | None = None, a3: float | None = None):
        rotations = self.get_rotations(a1, a2, a3).round(9)
        unique_rotations = np.unique(rotations)
        num = len(unique_rotations)
        return np.sum(
            np.isclose(rotations, unique_rotations[:, np.newaxis]) * np.arange(num)[:, np.newaxis],
            axis=0,
        )

    def get_cells(self, a1: float, a2: float | None = None, a3: float | None = None):
        positions = self.get_positions(a1, a2, a3)
        rotations = self.get_rotations(a1, a2, a3)
        indices = self.get_indices(a1, a2, a3)
        return positions, rotations, indices

    def _get_sizes(self, a1: float, a2: float | None = None, a3: float | None = None):
        periods = _fill_periods(a1, a2, a3)[: len(self._vectors)]

        axes_contributions = self._size_factors * self._vectors.T * np.asarray(periods)
        sizes = axes_contributions.sum(axis=1)
        assert isinstance(sizes, np.ndarray)
        return sizes[0], sizes[1]

    def get_n_periods(self):
        return self._n_periods

    def get_n_units(self):
        return self._n_units


def _get_av(angle: float):
    """Produces vectors for an angle.
    Produces [1.0, 0.0] for 0 and [0.0, 1.0] for 90.
    """
    return np.round([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))], 5)


# Use angles between [0, 180]. This way 0 (horizontal) will always be first
# and those closest to 90 (vertical) will be last.

_a1, _a2 = 0.0, 90.0
_vectors = [_get_av(_a1), _get_av(_a2)]
_coefficients = [[0.0], [0.0]]
_rotations = [_a1]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
_n_periods, _n_units = (1, 1)
rectangular = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)


_a1, _a2 = 0.0, 90.0
_vectors = [_get_av(_a1), _get_av(_a2)]
_coefficients = [[0.5, 0.0], [0.0, 0.5]]
_rotations = [_a1, _a2]
_size_factors = [[1.0, 0.0], [0.0, 1.0]]
_n_periods, _n_units = (2, 2)
square_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)


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
_n_periods, _n_units = (1, 3)
trigonal_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)


_a1, _a2, _a3 = 0.0, 60.0, 120.0
_vectors = [_get_av(_a1), _get_av(_a2), _get_av(_a3)]
_coefficients = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, -0.5],
    [-0.5, -0.5, 0.0],
    [-0.5, 0.0, 0.5],
    [1.0, 1.0, 0.0],
]
_rotations = [_a1, _a2, _a3, _a2, _a3, _a1]
_size_factors = [[2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
_n_periods, _n_units = (1, 3)
kagome_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)

_a1, _a2 = 0.0, 90.0
_vectors = [_get_av(_a1), _get_av(_a2)]
_coefficients = [
    [-1.5, 0.0],
    [-0.5, 0.0],
    [0.5, 0.0],
    [1.5, 0.0],
    [0.5, -1.0],
    [1.5, -1.0],
    [-1.5, -2.0],
    [-0.5, -2.0],
    [0.5, -2.0],
    [1.5, -2.0],
    [-0.5, 1.0],
    [-1.5, 1.0],
    [-2.0, -1.5],
    [-2.0, -0.5],
    [-2.0, 0.5],
    [-2.0, 1.5],
    [-1.0, -1.5],
    [-1.0, -0.5],
    [0.0, -1.5],
    [0.0, -0.5],
    [0.0, 0.5],
    [0.0, 1.5],
    [1.0, 0.5],
    [1.0, 1.5],
]
_rotations = [*([_a1] * 4 * 3), *([_a2] * 4 * 3)]
_size_factors = [[4.0, 0.0], [0.0, 4.0]]
_n_periods, _n_units = (2, 2)
shakti_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)

_a1, _a2 = 0.0, 90.0
_vectors = [_get_av(_a1), _get_av(_a2)]
_coefficients = [
    [0.5, -4.0],
    [1.5, -4.0],
    [-0.5, -3.0],
    [1.5, -3.0],
    [-1.5, -2.0],
    [1.5, -2.0],
    [-1.5, -1.0],
    [0.5, -1.0],
    [-1.5, 0.0],
    [-0.5, 0.0],
    [-0.5, 1.0],
    [1.5, 1.0],
    [-0.5, 2.0],
    [0.5, 2.0],
    [-1.5, 3.0],
    [0.5, 3.0],
    [-2.0, -2.5],
    [-2.0, -1.5],
    [-2.0, 0.5],
    [-2.0, 1.5],
    [-2.0, 2.5],
    [-2.0, 3.5],
    [-1.0, -3.5],
    [-1.0, -2.5],
    [-1.0, -0.5],
    [-1.0, 0.5],
    [-1.0, 2.5],
    [-1.0, 3.5],
    [0.0, -3.5],
    [0.0, -2.5],
    [0.0, -1.5],
    [0.0, -0.5],
    [0.0, 1.5],
    [0.0, 2.5],
    [1.0, -3.5],
    [1.0, -1.5],
    [1.0, -0.5],
    [1.0, 0.5],
    [1.0, 1.5],
    [1.0, 3.5],
]
_rotations = [*([_a1] * 8 * 2), *([_a2] * 4 * 6)]
_size_factors = [[4.0, 0.0], [0.0, 8.0]]
_n_periods, _n_units = (2, 2)
tetris_spin_ice = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)

_a1, _a2, _a3 = 0.0, 60.0, 120.0
_vectors = [_get_av(_a1), _get_av(_a2), _get_av(_a3)]
_coefficients = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]
_rotations = [_a1] * 2
_size_factors = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
_n_periods, _n_units = (1, 1)
rhombic = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)


_a1, _a2, _a3 = 0.0, 60.0, 120.0
_vectors = [_get_av(_a1), _get_av(_a2), _get_av(_a3)]
_coefficients = [
    [0.5, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [-0.5, 0.0, 0.0],
    [-0.5, -1.0, 0.0],
]
_rotations = [_a1] * 4
_size_factors = [[2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
_n_periods, _n_units = (1, 1)
honeycomb = Tiling(_vectors, _coefficients, _rotations, _size_factors, _n_periods, _n_units)
