from collections.abc import Sequence
from numbers import Real

import numpy as np

import mask.tiling.tiling as tiling

from .shape import Shape, empty_shape
from .utils import single_to_tuple


def square_spin_ice(units: Shape | Sequence[Shape], periods: float | Sequence[float]):
    units = single_to_tuple(units, Shape, 3)
    a1, a2, a3 = single_to_tuple(periods, Real, 3)

    tile = tiling.square_spin_ice
    rotations = tile.get_rotations(a1, a2, a3)  # type: ignore
    positions = tile.get_positions(a1, a2, a3)  # type: ignore
    extent = tile.get_extent(a1, a2, a3)  # type: ignore
    unique_rotations = np.unique(np.array(rotations))
    array = empty_shape
    for rot, pos in zip(rotations, positions):
        index: int = np.where(unique_rotations == rot)[0][0]
        array = array.union(units[index].translate(*pos).rotate(rot))
    return array.wrap_inside_extent(extent)
