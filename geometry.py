import numpy as np
from numpy import ndarray

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


def complex2cart(cnum: complex) -> tuple[float, float]:
    """"""
    return cnum.real, cnum.imag


def conjugate(points: ndarray) -> ndarray:
    """Return the complex conjugate. This is equivalent to a reflection across the real axis."""
    # return points.__class__(p.conjugate() for p in points)
    return np.vectorize(complex.conjugate)(points)


def centroid(points: ndarray) -> complex:
    """Compute the centroid of a list of points."""
    return np.sum(points) / len(points)


def translate(points: ndarray, amount: float) -> ndarray:
    """Translate a set of points by a constant amount."""
    return points + amount


def rotate(points: ndarray, theta: float, origin: complex = 0j) -> ndarray:
    """Rotate a set of points by theta radians. Optionally provide an origin to rotate around."""
    return origin + np.exp(1j * theta) * (points - origin)


def scale(points: ndarray, k: float, origin: complex = 0j) -> ndarray:
    """Zoom/Scale a set of points relative to an origin."""
    return k * (points - origin) + origin


class Polygon(ndarray):
    """"""

    id: str = "Polygon"

    def __new__(cls, input_array: ndarray, id=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.id = id
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.id = getattr(obj, "id", None)
