from typing import Iterable
import numpy as np

# A small tolerance for comparing floats for equality
RTOL = 1.0e-5
ATOL = 1.0e-8
TOL = 1.0e-5

# psi = 1/phi where phi is the Golden ratio, sqrt(5)+1)/2
PSI: float = (np.sqrt(5.0) - 1.0) / 2.0
# psi**2 = 1 - psi
PSI2: float = 1 - PSI


def isclose(a, b):
    """Wrapper for numpy isclose."""
    return np.isclose(a, b, RTOL, ATOL, False)


def cart2complex(*coord: float) -> complex:
    """Return x, y coordinates as a + bj complex number."""
    a, b = coord
    return a + 1j * b


def complex2cart(cnum: complex) -> tuple[float, float]:
    """"""
    return cnum.real, cnum.imag


def conjugate(points: Iterable[complex]) -> Iterable[complex]:
    """Return the complex conjugate. This is equivalent to a reflection across the real axis."""
    return points.__class__(p.conjugate() for p in points)


def centroid(points: Iterable[complex]) -> complex:
    """Compute the centroid of a list of points."""
    return np.sum(points) / len(points)


def translate(points: Iterable[complex], amount: float) -> Iterable[complex]:
    """Translate a set of points by a constant amount."""
    return points.__class__(p + amount for p in points)


def rotate(
    points: Iterable[complex], theta: float, origin: complex = 0j
) -> Iterable[complex]:
    """Rotate a set of points by theta radians. Optionally provide an origin to rotate around."""
    return points.__class__(origin + np.exp(1j * theta) * (p - origin) for p in points)


def scale(
    points: Iterable[complex], k: float, origin: complex = 0j
) -> Iterable[complex]:
    """Zoom/Scale a set of points relative to an origin."""
    return points.__class__(k * (p - origin) for p in points)


class Polygon(tuple):
    """"""
