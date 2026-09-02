from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real
from typing import cast

import numpy as np

import mask.tiling.tiling as tiling

from .shape import Shape, empty_shape
from .utils import single_to_tuple

UnitsArg = Shape | Sequence[Shape]
PeriodsArg = float | Sequence[float]
TileFunc = Callable[[UnitsArg, PeriodsArg], Shape]


def _build(tile: tiling.Tiling, units: UnitsArg, periods: PeriodsArg) -> Shape:
    # Handle excess / missing units and periods
    build_units = single_to_tuple(units, Shape, tile.get_n_units())
    build_periods = single_to_tuple(periods, Real, tile.get_n_periods())
    build_periods = cast(Sequence[float], build_periods)  # type: ignore

    positions, rotations, indices = tile.get_cells(*build_periods)
    n_unique_rotations = len(np.unique(indices))
    if n_unique_rotations != len(build_units):
        raise ValueError(
            f"Tiling has {n_unique_rotations} distinct orientations "
            f"but {len(build_units)} unit shapes were given."
        )

    array = empty_shape
    for p, r, i in zip(positions, rotations, indices):
        array = array.union(build_units[i].translate(*p).rotate(r))
    return array.wrap_inside_extent(tile.get_extent(*build_periods))


def square_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Square spin ice array."""
    return _build(tiling.square_spin_ice, units, periods)


def trigonal_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Trigonal spin ice array."""
    return _build(tiling.trigonal_spin_ice, units, periods)


def kagome_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Kagome spin ice array."""
    return _build(tiling.kagome_spin_ice, units, periods)


def shakti_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Shakti spin ice array."""
    return _build(tiling.shakti_spin_ice, units, periods)


def tetris_spin_ice(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Tetris spin ice array."""
    return _build(tiling.tetris_spin_ice, units, periods)


def rectangular(units: UnitsArg, periods: PeriodsArg) -> Shape:
    """Rectangular tiling."""
    return _build(tiling.rectangular, units, periods)
