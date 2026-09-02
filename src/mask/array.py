from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real
from typing import TypedDict, cast

import numpy as np
from typing_extensions import Unpack

import mask.tiling.tiling as tiling

from .shape import Shape, empty_shape
from .utils import single_to_tuple

UnitsArg = Shape | Sequence[Shape]
PeriodsArg = float | Sequence[float]
TileFunc = Callable[[UnitsArg, PeriodsArg], Shape]


class BuildOptions(TypedDict, total=False):
    nx: int
    ny: int
    rotation: float
    view_offset: tuple[float, float]


def _build_cell(
    tile: tiling.Tiling,
    units: UnitsArg,
    periods: PeriodsArg,
    rotation: float,
    view_offset: tuple[float, float],
):
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
    extent = tile.get_extent(*build_periods)
    # Handle view_offset and rotation
    positions += (view_offset[0] * extent.get_width(), view_offset[1] * extent.get_height())
    rotations += rotation

    array = empty_shape.union(
        [build_units[i].translate(*p).rotate(r) for p, r, i in zip(positions, rotations, indices)]
    )
    return array, extent
    # return array.wrap_inside_extent(extent), extent


def _build(
    tile: tiling.Tiling,
    units: UnitsArg,
    periods: PeriodsArg,
    *,
    nx: int = 1,
    ny: int = 1,
    rotation: float = 0.0,
    view_offset: tuple[float, float] = (0.0, 0.0),
) -> Shape:
    cell, cell_extent = _build_cell(tile, units, periods, rotation, view_offset)
    w, h = cell_extent.get_sizes()
    array = empty_shape.union([cell.translate(i * w, j * h) for i in range(nx) for j in range(ny)])
    array = array.wrap_inside_extent(cell_extent.scale((nx, ny)))
    # print(len(array.to_shapely().geoms))
    return array


def square_spin_ice(units: UnitsArg, periods: PeriodsArg, **options: Unpack[BuildOptions]) -> Shape:
    """Square spin ice array."""
    return _build(tiling.square_spin_ice, units, periods, **options)


def trigonal_spin_ice(
    units: UnitsArg, periods: PeriodsArg, **options: Unpack[BuildOptions]
) -> Shape:
    """Trigonal spin ice array."""
    return _build(tiling.trigonal_spin_ice, units, periods, **options)


def kagome_spin_ice(units: UnitsArg, periods: PeriodsArg, **options: Unpack[BuildOptions]) -> Shape:
    """Kagome spin ice array."""
    return _build(tiling.kagome_spin_ice, units, periods, **options)


def shakti_spin_ice(units: UnitsArg, periods: PeriodsArg, **options: Unpack[BuildOptions]) -> Shape:
    """Shakti spin ice array."""
    return _build(tiling.shakti_spin_ice, units, periods, **options)


def tetris_spin_ice(units: UnitsArg, periods: PeriodsArg, **options: Unpack[BuildOptions]) -> Shape:
    """Tetris spin ice array."""
    return _build(tiling.tetris_spin_ice, units, periods, **options)


def rectangular(units: UnitsArg, periods: PeriodsArg, **options: Unpack[BuildOptions]) -> Shape:
    """Rectangular tiling."""
    return _build(tiling.rectangular, units, periods, **options)
