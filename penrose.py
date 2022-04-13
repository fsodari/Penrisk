from pathlib import Path
from typing import Iterable, Protocol
import numpy as np

from geometry import Polygon, isclose, PSI, PSI2
from geometry import centroid, rotate


class RobinsonTriangle(Polygon):
    """A rhombus created from reflecting an isosceles triangle."""

    def __new__(cls, points: Iterable[complex]):
        # This shape can be initialized with 3 or 4 points. If 4 points are provided, the 4th point is ignored and re-generated.
        a, b, c, *_ = points

        # Check if ab == bc
        if not isclose(abs(b - a), abs(c - b)):
            raise Exception("Must initialze RobinsonTriangle with isosceles triangle.")

        origin = (a + c) / 2.0
        d = rotate((b,), np.pi, origin)

        return super(RobinsonTriangle, cls).__new__(cls, (a, b, c, *d))


class FatRhombus(RobinsonTriangle):
    def inflate(self) -> Iterable[RobinsonTriangle]:
        """"""
        # D and E divide sides AC and AB respectively
        a, b, c, _ = self

        d = PSI2 * a + PSI * c
        e = PSI2 * a + PSI * b
        # Take care to order the vertices here so as to get the right
        # orientation for the resulting triangles.
        return (
            FatRhombus((d, e, a)),
            ThinRhombus((e, d, b)),
            FatRhombus((c, d, b)),
        )


class ThinRhombus(RobinsonTriangle):
    """"""

    def inflate(self) -> Iterable[RobinsonTriangle]:
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.
        """
        a, b, c, _ = self

        d = PSI * a + PSI2 * b
        return ThinRhombus((d, c, a)), FatRhombus((c, d, b))


class SupportsInflate(Protocol):
    def inflate(self) -> Iterable[RobinsonTriangle]:
        ...


def inflate(triangle: SupportsInflate) -> Iterable[RobinsonTriangle]:
    return triangle.inflate()


def svg_path(points: Iterable[complex]):
    """
    Return the SVG "d" path element specifier for a polygon. Make sure the points are in the right order.
    """

    svg_str = ""
    # Make a list we can
    # pts = [p for p in points]

    for i, p in enumerate(points):
        if i == 0:
            svg_str += f"m{p.real}, {p.imag}"
        else:
            diff = p - p_prev
            svg_str += f" l{diff.real}, {diff.imag}"
        p_prev = p

    svg_str += "z"
    return svg_str

    # ab, bc, cd = self.b - self.a, self.c - self.b, self.d - self.c
    # xy = lambda v: (v.real, v.imag)
    # if rhombus:
    #     return "m{},{} l{},{} l{},{} l{},{}z".format(
    #         *xy(self.a) + xy(ab) + xy(bc) + xy(cd)
    #     )
    # return "m{},{} l{},{} l{},{}z".format(*xy(self.a) + xy(ab) + xy(bc))


def remove_dupes(tiles: list[RobinsonTriangle]):
    """
    Remove tiles giving rise to identical rhombuses from the
    ensemble.
    """

    # tiles give rise to identical rhombuses if these rhombuses have
    # the same centre.
    selements = sorted(tiles, key=lambda e: (centroid(e).real, centroid(e).imag))
    elements = [selements[0]]
    for i, element in enumerate(selements[1:], start=1):
        if not isclose(centroid(element), centroid(selements[i - 1])):
            elements.append(element)

    return elements


def is_in_box(
    triangle: RobinsonTriangle,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    margin: float = 1.0,
) -> bool:
    """Returns true if the center of the triangle is inside the bounding box."""
    c = centroid(triangle)

    x_marg_amt = (margin - 1.0) * (xbound[1] - xbound[0])
    y_marg_amt = (margin - 1.0) * (ybound[1] - ybound[0])

    return (
        c.real >= (xbound[0] - x_marg_amt)
        and c.real <= (xbound[1] + x_marg_amt)
        and c.imag >= (ybound[0] - y_marg_amt)
        and c.imag <= (ybound[1] + y_marg_amt)
    )


def find_minmax(tiling: list[RobinsonTriangle]) -> tuple[complex, ...]:
    minx = centroid(tiling[0])
    maxx = minx
    miny = minx
    maxy = miny

    for t in tiling:
        c = centroid(t)

        if c.real < minx.real:
            minx = c
        if c.real > maxx.real:
            maxx = c
        if c.imag < miny.imag:
            miny = c
        if c.imag > maxy.imag:
            maxy = c

    return minx, maxx, miny, maxy


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
                svg_path(t),
            )
        )

    svg.append("</g>\n</svg>")

    return "\n".join(svg)


def write_svg(svg, file: Path):
    with open(file, "w") as f:
        f.write(svg)


if __name__ == "__main__":
    pts = 1j, 0, 1.0
    x = rotate(pts, np.pi / 10.0)
    tri = RobinsonTriangle(x)
    print(tri)
