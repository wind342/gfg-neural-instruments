class MeasurementError(ValueError):
    """A protocol or identity violation; never silently converted to a number."""


def require(condition, code):
    if not condition:
        raise MeasurementError(code)
