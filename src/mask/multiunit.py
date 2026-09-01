from .shape import Shape


def logic_dots(unit: Shape, period: float | tuple[float, float], ndots: int):
    period = (period, period) if isinstance(period, float | int) else period
    translations = [(0.0, -period[1]), (period[0], 0.0), (0.0, period[1]), (-period[0], 0.0)]

    if ndots < 1:
        ndots = 1

    logic_dots = unit.translate(0.0, 0.0)
    for i in range(ndots - 1):
        logic_dots = logic_dots.union(unit.translate(*translations[i]))

    return logic_dots.centre()
