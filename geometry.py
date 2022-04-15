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


def intersects(poly1: Polygon, poly2: Polygon) -> bool:
    """
    Check if polygon intersects with another Polygon

    Compare each edge of each polygon with each edge of the other polygon.

    1. Construct a unit normal vector to the edge.

    2. Take the dot product of each polygon's point with this vector.

    3. Compare the min/max of each polygon to see if they overlap.

    """

    # Check all of poly1's edges.
    # for i, p in enumerate(poly1):
    #     p: complex
    #     edge = p - poly1[i - 1]
    #     # perpindicular to the edge
    #     perp = complex(-1 * edge.imag, edge.real)
    #     # norm = perp / np.abs(edge)
    #     # print(np.abs(norm))

    #     # Project polygon onto normal vector axis
    #     poly1_proj = [np.vdot(perp, pt) for pt in poly1]
    #     poly2_proj = [np.vdot(perp, pt) for pt in poly2]

    #     # print(poly1_proj)
    #     # print(poly2_proj)

    #     if np.min(poly1_proj) > np.max(poly2_proj):
    #         return True

    #     if np.max(poly1_proj) < np.min(poly2_proj):
    #         return True

    # Check all of poly2's edges.
    for i, p in enumerate(poly1):
        p: complex

        x1, x2 = poly1[i - 1].real, p.real
        y1, y2 = poly1[i - 1].imag, p.imag

        for k, p2 in enumerate(poly2):
            x3, x4 = poly2[k - 1].real, p2.real
            y3, y4 = poly2[k - 1].imag, p2.imag

            num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            t = num / den

            num = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            u = num / den

            if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                return True

        # edge = p - poly2[i - 1]
        # # perpindicular to the edge
        # perp = rotate(edge, np.pi / 2)
        # # norm = perp / np.abs(edge)

        # # Project polygon onto normal vector axis
        # poly1_proj = [np.dot(perp, pt) for pt in poly1]
        # poly2_proj = [np.dot(perp, pt) for pt in poly2]

        # if np.min(poly1_proj) > np.max(poly2_proj):
        #     return True

        # if np.max(poly1_proj) < np.min(poly2_proj):
        #     return True

    return False

    # def ccw(a: complex, b: complex, c: complex):
    #     return (c.imag - a.imag) * (b.real - a.real) > (b.imag - a.imag) * (
    #         c.real - a.real
    #     )

    # # Check each edge in the polygon with each line
    # for i, p in enumerate(points):
    #     for l in lines:
    #         a, b = points[i - 1], p
    #         c, d = l[0], l[1]

    #         ix = ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    #         if ix:
    #             return True

    # return False
