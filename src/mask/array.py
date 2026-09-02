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
    # print(len(positions))
    for rot, pos in zip(rotations, positions):
        # print(rot, pos)
        index = int(np.where(unique_rotations == rot)[0][0])
        array = array.union(units[index].translate(*pos).rotate(rot))
    return array.wrap_inside_extent(extent)


def square_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Square spin ice array."""
    return _build(
        tiling.square_spin_ice,
        single_to_tuple(units, Shape, 2),
        single_to_tuple(periods, Real, 2),  # type: ignore
    )


def trigonal_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Trigonal spin ice array."""
    return _build(
        tiling.trigonal_spin_ice,
        single_to_tuple(units, Shape, 3),
        single_to_tuple(periods, Real, 1),  # type: ignore
    )


def kagome_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Kagome spin ice array."""
    return _build(
        tiling.kagome_spin_ice,
        single_to_tuple(units, Shape, 3),
        single_to_tuple(periods, Real, 1),  # type: ignore
    )


def shakti_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Shakti spin ice array."""
    return _build(
        tiling.square_spin_ice,
        single_to_tuple(units, Shape, 2),
        single_to_tuple(periods, Real, 2),  # type: ignore
    )


def rectangular(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Rectangular tiling."""
    return _build(
        tiling.rectangular,
        single_to_tuple(units, Shape, 1),
        single_to_tuple(periods, Real, 1),  # type: ignore
    )
