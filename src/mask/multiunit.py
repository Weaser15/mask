from .shape import Shape, empty_shape


def logic_dots(unit: Shape, period: float | tuple[float, float], ndots: int):
    period = (period, period) if isinstance(period, float | int) else period
    translations = [(0.0, -period[1]), (period[0], 0.0), (0.0, period[1]), (-period[0], 0.0)]

    if ndots < 1:
        ndots = 1

    logic_dots = unit.translate(0.0, 0.0)
    for i in range(ndots - 1):
        logic_dots = logic_dots.union(unit.translate(*translations[i]))

    return logic_dots.centre()


def spiral(unit1: Shape, unit2: Shape, spacing: float | tuple[float, float]):
    spacing = (spacing, spacing) if isinstance(spacing, float | int) else spacing

    u1_width, u1_height = unit1.get_extent().get_sizes()
    u2_width, u2_height = unit2.get_extent().get_sizes()

    x_unit = 0.5 * (u1_height + spacing[0])
    y_unit = 0.5 * (u2_height + spacing[1])

    x_unit_2 = 0.5 * spacing[0] + 0.5 * u2_width
    y_unit_2 = 0.5 * spacing[1] + 0.5 * u1_width

    translations = [
        (x_unit_2, -y_unit),
        (x_unit, y_unit_2),
        (-x_unit_2, y_unit),
        (-x_unit, -y_unit_2),
    ]

    spiral = empty_shape
    rotations = [90.0, 0.0, 90.0, 0.0]
    for i, (translation, rotation) in enumerate(zip(translations, rotations)):
        unit = unit1 if i % 2 == 0 else unit2
        spiral = spiral.union(unit.translate(*translation).rotate(rotation))
    return spiral
