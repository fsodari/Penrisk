from pathlib import Path
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

# A small tolerance for comparing floats for equality
TOL = 1.0e-5
# psi = 1/phi where phi is the Golden ratio, sqrt(5)+1)/2
PSI = (np.sqrt(5) - 1) / 2
# psi**2 = 1 - psi
PSI2 = 1 - PSI


def cart2complex(*coord):
    """Return x, y coordinates as a + bj complex number."""
    return coord[0] + 1j * coord[1]


def point(*args):
    return cart2complex(*args)


@dataclass
class RobinsonTriangle:
    a: complex
    b: complex
    c: complex

    def center(self):
        return (self.a + self.c) / 2.0

    def conjugate(self):
        return self.__class__(
            self.a.conjugate(), self.b.conjugate(), self.c.conjugate()
        )

    def rotate(self, theta: float):
        """"""
        rot = np.cos(theta) + 1j * np.sin(theta)
        a = self.a * rot
        b = self.b * rot
        c = self.c * rot

        return self.__class__(a, b, c)

    def translate(self, amount: complex):
        a = self.a + amount
        b = self.b + amount
        c = self.c + amount

        return self.__class__(a, b, c)

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
    def inflate(self):
        pass


@dataclass
class FatRhombus(RobinsonTriangle):
    """"""

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
    """"""

    def inflate(self) -> list[RobinsonTriangle]:
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.
        """
        d = PSI * self.a + PSI2 * self.b
        return [SkinnyRhombus(d, self.c, self.a), FatRhombus(self.c, d, self.b)]


def translate(*vertices: complex, amount: complex):
    return [v + amount for v in vertices]


def remove_dupes(tiles):
    """
    Remove tiles giving rise to identical rhombuses from the
    ensemble.
    """

    # tiles give rise to identical rhombuses if these rhombuses have
    # the same centre.
    selements: list[RobinsonTriangle] = sorted(
        tiles, key=lambda e: (e.center().real, e.center().imag)
    )
    elements = [selements[0]]
    for i, element in enumerate(selements[1:], start=1):
        element: RobinsonTriangle
        if abs(element.center() - selements[i - 1].center()) > TOL:
            elements.append(element)

    return elements


def all_points(shapes: list[RobinsonTriangle]):
    """Get all points from shapes. Remove duplicate points"""


def make_svg(
    tiling: list[RobinsonTriangle],
    minx: float = 0,
    maxx: float = 800,
    miny: float = 0,
    maxy: float = 800,
    stroke_width: float = 0.01,
    draw_rhombuses: bool = True,
):
    """Make and return the SVG for the tiling as a str."""

    stroke_color = "#000000"

    viewbox = f"{minx} {miny} {(maxx-minx)} {maxy-miny}"
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
