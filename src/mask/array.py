from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real

import numpy as np

import mask.tiling.tiling as tiling

from .shape import Shape, empty_shape
from .utils import single_to_tuple

UnitsArg = Shape | Sequence[Shape]
PeriodsArg = float | Sequence[float]
TileFunc = Callable[[UnitsArg, PeriodsArg], Shape]


def _build(tile: tiling.Tiling, units: Sequence[Shape], periods: Sequence[float]) -> Shape:
    rotations = tile.get_rotations(*periods)
    positions = tile.get_positions(*periods)
    extent = tile.get_extent(*periods)

    unique_rotations = np.unique(np.asarray(rotations))
    if len(unique_rotations) != len(units):
        raise ValueError(
            f"Tiling has {len(unique_rotations)} distinct orientations "
            f"but {len(units)} unit shapes were given."
        )

    array = empty_shape
    for rot, pos in zip(rotations, positions):
        index = int(np.where(unique_rotations == rot)[0][0])
        array = array.union(units[index].translate(*pos).rotate(rot))
    return array.wrap_inside_extent(extent)


def square_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Square spin ice array."""
    num = 2
    return _build(
        tiling.square_spin_ice,
        single_to_tuple(units, Shape, num),
        single_to_tuple(periods, Real, num),  # type: ignore
    )


def rectangular(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Rectangular tiling."""
    num = 1
    return _build(
        tiling.rectangular,
        single_to_tuple(units, Shape, num),
        single_to_tuple(periods, Real, num),  # type: ignore
    )
