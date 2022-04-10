from pathlib import Path
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

# A small tolerance for comparing floats for equality
TOL = 1.0e-5
# psi = 1/phi where phi is the Golden ratio, sqrt(5)+1)/2
PSI: float = (np.sqrt(5.0) - 1.0) / 2.0
# psi**2 = 1 - psi
PSI2: float = 1 - PSI


def cart2complex(*coord):
    """Return x, y coordinates as a + bj complex number."""
    a, b = coord
    return a + 1j * b


def complex2cart(cnum: complex) -> tuple[float, float]:
    """"""
    return cnum.real, cnum.imag


def point(*args):
    """Create a complex number from cartesian coordinates."""
    return cart2complex(*args)


class RobinsonTriangle:
    """"""


@dataclass
class RobinsonTriangle:
    a: complex
    b: complex
    c: complex

    def path(self, rhombus=True):
        """
        Return the SVG "d" path element specifier for the rhombus formed
        by this triangle and its mirror image joined along their bases. If
        rhombus=False, the path for the triangle itself is returned instead.
        """

        ab, bc = self.b - self.a, self.c - self.b
        xy = lambda v: (v.real, v.imag)
        if rhombus:
            return "m{},{} l{},{} l{},{} l{},{}z".format(
                *xy(self.a) + xy(ab) + xy(bc) + xy(-ab)
            )
        return "m{},{} l{},{} l{},{}z".format(*xy(self.a) + xy(ab) + xy(bc))

    @abstractmethod
    def inflate(self) -> list[RobinsonTriangle]:
        """Run the inflation procedure to generate penrose tiles. Each shape has a different inflation process."""
        pass


@dataclass
class FatRhombus(RobinsonTriangle):
    """
    This triangle has side lengths with the ratio 1 : 1 : phi
    """

    def inflate(self) -> list[RobinsonTriangle]:
        """"""
        # D and E divide sides AC and AB respectively
        d = PSI2 * self.a + PSI * self.c
        e = PSI2 * self.a + PSI * self.b
        # Take care to order the vertices here so as to get the right
        # orientation for the resulting triangles.
        return [
            FatRhombus(d, e, self.a),
            SkinnyRhombus(e, d, self.b),
            FatRhombus(self.c, d, self.b),
        ]


@dataclass
class SkinnyRhombus(RobinsonTriangle):
    """This triangle has side lengths with the ratio 1 : 1 : psi"""

    def inflate(self) -> list[RobinsonTriangle]:
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.
        """
        d = PSI * self.a + PSI2 * self.b
        return [SkinnyRhombus(d, self.c, self.a), FatRhombus(self.c, d, self.b)]


def center(triangle: RobinsonTriangle) -> complex:
    """Return the center of a robinson triangle. This is the midpoint of the base of an isosceles triangle."""
    return (triangle.a + triangle.c) / 2.0


def conjugate(triangle: RobinsonTriangle) -> RobinsonTriangle:
    """Returns a new triangle made from the complex conjugate of the vertices. This is equivalent to a reflection."""
    return triangle.__class__(
        triangle.a.conjugate(), triangle.b.conjugate(), triangle.c.conjugate()
    )


def rotate(
    triangle: RobinsonTriangle, theta: float, origin: complex = 0 + 0j
) -> RobinsonTriangle:
    """
    Performs a rotation about an origin and returns the new triangle.
    If no origin is given, 0 + 0j will be used.
    Rotation Equation: a' = origin + e**(j*theta)(a - origin)
    """
    a = origin + np.exp(1j * theta) * (triangle.a - origin)
    b = origin + np.exp(1j * theta) * (triangle.b - origin)
    c = origin + np.exp(1j * theta) * (triangle.c - origin)

    # Return a new triangle
    return triangle.__class__(a, b, c)


def translate(triangle: RobinsonTriangle, amount: complex = 0 + 0j) -> RobinsonTriangle:
    """Translates by a given amount and returns a new triangle."""
    a = triangle.a + amount
    b = triangle.b + amount
    c = triangle.c + amount

    # Return a new triangle
    return triangle.__class__(a, b, c)


def inflate(triangle: RobinsonTriangle) -> list[RobinsonTriangle]:
    """"""
    return triangle.inflate()


def remove_dupes(tiles: list[RobinsonTriangle]):
    """
    Remove tiles giving rise to identical rhombuses from the
    ensemble.
    """

    # tiles give rise to identical rhombuses if these rhombuses have
    # the same centre.
    selements = sorted(tiles, key=lambda e: (center(e).real, center(e).imag))
    elements = [selements[0]]
    for i, element in enumerate(selements[1:], start=1):
        if abs(center(element) - center(selements[i - 1])) > TOL:
            elements.append(element)

    return elements


def all_points(shapes: list[RobinsonTriangle]):
    """Get all points from shapes. Remove duplicate points"""


def is_in_box(
    triangle: RobinsonTriangle,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    margin: float = 1.0,
) -> bool:
    """Returns true if the center of the triangle is inside the bounding box."""
    c = center(triangle)

    x_marg_amt = (margin - 1.0) * (xbound[1] - xbound[0])
    y_marg_amt = (margin - 1.0) * (ybound[1] - ybound[0])

    return (
        c.real >= (xbound[0] - x_marg_amt)
        and c.real <= (xbound[1] + x_marg_amt)
        and c.imag >= (ybound[0] - y_marg_amt)
        and c.imag <= (ybound[1] + y_marg_amt)
    )


def find_minmax(tiling: list[RobinsonTriangle]) -> tuple[complex]:
    minx = tiling[0].a.real
    maxx = minx
    miny = tiling[0].a.imag
    maxy = miny

    for t in tiling:
        new_minx = np.min([t.a.real, t.b.real, t.c.real])
        new_maxx = np.max([t.a.real, t.b.real, t.c.real])
        new_miny = np.min([t.a.imag, t.b.imag, t.c.imag])
        new_maxy = np.max([t.a.imag, t.b.imag, t.c.imag])

        if new_minx < minx:
            minx = new_minx
        if new_maxx > maxx:
            maxx = new_maxx
        if new_miny < miny:
            miny = new_miny
        if new_maxy > maxy:
            maxy = new_maxy

    return minx + 0j, maxx + 0j, 1j * miny, 1j * maxy


def make_svg(
    tiling: list[RobinsonTriangle],
    stroke_width: float = 0.01,
    draw_rhombuses: bool = True,
):
    """Make and return the SVG for the tiling as a str."""

    minx, maxx, miny, maxy = find_minmax(tiling)
    stroke_color = "#000000"

    viewbox = f"{minx.real} {miny.imag} {(maxx-minx).real} {(maxy-miny).imag}"

    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg width="100%" height="100%" viewBox="{viewbox}"'
        ' preserveAspectRatio="xMidYMid meet" version="1.1"'
        ' baseProfile="full" xmlns="http://www.w3.org/2000/svg">',
    ]

    # The tiles' stroke widths scale with ngen
    svg.append(
        '<g style="stroke:{}; stroke-width: {};'
        ' stroke-linejoin: round;">'.format(stroke_color, stroke_width)
    )

    for t in tiling:
        svg.append(
            '<path fill="#ffffff" fill-opacity="0.0" d="{}"/>'.format(
                t.path(rhombus=draw_rhombuses),
            )
        )

    svg.append("</g>\n</svg>")

    return "\n".join(svg)


def write_svg(svg, file: Path):
    with open(file, "w") as f:
        f.write(svg)
