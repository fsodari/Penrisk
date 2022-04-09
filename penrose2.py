from pathlib import Path
from sys import getallocatedblocks
from typing import Any, Iterable
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

# A small tolerance for comparing floats for equality
TOL = 1.0e-5
# psi = 1/phi where phi is the Golden ratio, sqrt(5)+1)/2
psi = (np.sqrt(5) - 1) / 2
# psi**2 = 1 - psi
psi2 = 1 - psi


def cart2complex(*coord):
    """Return x, y coordinates as a + bj complex number."""
    return coord[0] + 1j * coord[1]


def point(*args):
    return cart2complex(*args)


def triangle(a, b, c):
    return a, b, c


@dataclass
class RobinsonTriangle:
    """"""


@dataclass
class RobinsonTriangle:
    a: complex
    b: complex
    c: complex

    def center(self):
        return (self.a + self.c) / 2.0

    def conjugate(self):
        return RobinsonTriangle(
            self.a.conjugate(), self.b.conjugate(), self.c.conjugate()
        )

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

    def inflate(self):
        """"""
        # D and E divide sides AC and AB respectively
        d = psi2 * self.a + psi * self.c
        e = psi2 * self.a + psi * self.b
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

    def inflate(self):
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.
        """
        d = psi * self.a + psi2 * self.b
        return [SkinnyRhombus(d, self.c, self.a), FatRhombus(self.c, d, self.b)]


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


def make_svg(
    tiling: list[RobinsonTriangle],
    ngen: int,
    scale: float = 100.0,
    margin: float = 1.05,
    width: float = 800,
    height: float = 600,
    stroke_width: float = 0.01,
    draw_rhombuses: bool = True,
    image=None,
):
    """Make and return the SVG for the tiling as a str."""

    stroke_color = "#000000"

    stroke_width = str(psi**ngen * scale * stroke_width)

    xmin = ymin = -scale * margin
    # width = height = 2 * scale * margin
    viewbox = f"{xmin} {ymin} {width} {height}"
    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<svg width="{:.0f}%" height="{:.0f}" viewBox="{}"'
        ' preserveAspectRatio="xMidYMid meet" version="1.1"'
        ' baseProfile="full" xmlns="http://www.w3.org/2000/svg">'.format(
            100.0, 100.0, viewbox
        ),
    ]

    if image is not None:
        """"""
        img_tag = (
            f'<image xlink:href="{image}" x="0" y="0" width="100%" height="100%" />'
        )
        svg.append(img_tag)

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


def inf():
    """"""
    N = 8
    scale = 100

    theta = 2 * np.pi / 5
    rot = np.cos(theta) + 1j * np.sin(theta)

    a = point(-scale / 2, 0)
    b = scale / 2 * rot
    c = point(scale / 2 / psi, 0)

    tri = FatRhombus(a, b, c)
    tiling: list[RobinsonTriangle] = [tri]

    # Inflate N times. Remove duplicates after each generation.
    for _ in range(N):
        inflated = []
        for t in tiling:
            inflated.extend(t.inflate())
        tiling = inflated

    tiling = remove_dupes(tiling)

    # Add conjugate elements to reflect across the x axis
    conj = [t.conjugate() for t in tiling]
    tiling = remove_dupes(tiling + conj)

    # Make svg
    svg = make_svg(
        tiling,
        N,
        scale,
        width=1080,
        height=600,
        stroke_width=0.01,
        image=Path("map.png").absolute(),
    )
    write_svg(svg, "foo.svg")


if __name__ == "__main__":
    inf()
