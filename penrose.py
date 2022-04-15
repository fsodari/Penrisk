from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol
import numpy as np
from numpy import ndarray

from geometry import Polygon, intersects, isclose, PSI, PSI2, translate
from geometry import centroid, conjugate, rotate, scale

import cv2

# class RobinsonTriangle(Polygon):
#     """A rhombus created from reflecting an isosceles triangle."""

#     theta: float

#     def __new__(cls, points: ndarray):
#         # This shape can be initialized with 3 or 4 points. If 4 points are provided, the 4th point is ignored and re-generated.
#         a, b, c, *_ = points

#         # Check if ab == bc
#         # if not isclose(abs(b - a), abs(c - b)):
#         #     print(cls)
#         #     print(np.abs(b - a), np.abs(c - b), np.abs(a - c))
#         #     raise Exception("Must initialze RobinsonTriangle with isosceles triangle.")

#         origin = (a + c) / 2.0
#         d = rotate(b, np.pi, origin)

#         return super().__new__(cls, [a, b, c, d])


class RobinsonTriangle(Polygon):
    id: type[RobinsonTriangle]
    theta: float

    def __new__(cls, input_array: ndarray, id: type[RobinsonTriangle] = None):
        a, b, c, *_ = input_array

        # Check if ab == bc
        if not isclose(abs(b - a), abs(c - b)):
            print(cls)
            print(np.abs(b - a), np.abs(c - b), np.abs(a - c))
            raise Exception("Must initialze RobinsonTriangle with isosceles triangle.")

        origin = (a + c) / 2.0
        d = rotate(b, np.pi, origin)

        return super().__new__(cls, (a, b, c, d), id)

    def inflate(self) -> tuple[RobinsonTriangle, ...]:
        ...


class FatRhombus(RobinsonTriangle):
    theta: float = np.pi * 1.0 / 5.0

    def __new__(cls, input_array: ndarray):
        return super().__new__(cls, input_array, FatRhombus)

    @classmethod
    def inflate(cls, triangle: RobinsonTriangle) -> tuple[RobinsonTriangle, ...]:
        """"""
        # D and E divide sides AC and AB respectively
        a, b, c, *_ = triangle

        # Preserve side lengths
        scale_factor = 1.0 / PSI

        d = PSI2 * a + PSI * c
        e = PSI2 * a + PSI * b
        # Take care to order the vertices here so as to get the right
        # orientation for the resulting triangles.
        return (
            scale(FatRhombus((d, e, a)), scale_factor),
            scale(ThinRhombus((e, d, b)), scale_factor),
            scale(FatRhombus((c, d, b)), scale_factor),
        )


class ThinRhombus(RobinsonTriangle):
    # Using pi/2 - (1/5) * pi
    theta: float = np.pi * 2.0 / 5.0

    def __new__(cls, input_array: ndarray):
        return super().__new__(cls, input_array, ThinRhombus)

    @classmethod
    def inflate(cls, triangle: RobinsonTriangle) -> tuple[RobinsonTriangle, ...]:
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.
        """
        a, b, c, *_ = triangle
        d = PSI * a + PSI2 * b

        # Preserve side lengths
        scale_factor = 1.0 / PSI

        return (
            scale(ThinRhombus((d, c, a)), scale_factor),
            scale(FatRhombus((c, d, b)), scale_factor),
        )


def inflate(triangle: RobinsonTriangle) -> list[RobinsonTriangle]:
    return triangle.inflate(triangle)


def svg_path(points: ndarray):
    """
    Return the SVG "d" path element specifier for a polygon. Make sure the points are in the right order.
    """

    svg_str = ""
    for i, p in enumerate(points):
        p: complex
        if i == 0:
            svg_str += f"m{p.real}, {p.imag}"
        else:
            diff = p - p_prev
            svg_str += f" l{diff.real}, {diff.imag}"
        p_prev = p

    svg_str += "z"
    return svg_str


def remove_dupes(tiles: ndarray):
    """
    Remove tiles giving rise to identical rhombuses from the
    ensemble.
    """

    # tiles give rise to identical rhombuses if these rhombuses have
    # the same centre.
    selements = sorted(tiles, key=lambda e: (centroid(e).real, centroid(e).imag))
    elements = [selements[0]]
    for i, element in enumerate(selements[1:], start=1):
        # print(i)
        if not isclose(centroid(element), centroid(selements[i - 1])):
            elements.append(element)

    return elements


def is_in_box(
    triangle: RobinsonTriangle,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
) -> bool:
    """Returns true if the center of the triangle is inside the bounding box."""
    c = centroid(triangle)

    return (
        c.real >= xbound[0]
        and c.real <= xbound[1]
        and c.imag >= ybound[0]
        and c.imag <= ybound[1]
    )


def find_minmax(poly: Polygon) -> tuple[complex, ...]:
    minx = poly[0]
    maxx = minx
    miny = minx
    maxy = miny

    for p in poly:
        p: complex
        if p.real < minx.real:
            minx = p
        if p.real > maxx.real:
            maxx = p
        if p.imag < miny.imag:
            miny = p
        if p.imag > maxy.imag:
            maxy = p

    return minx, maxx, miny, maxy


def find_minmaxv(polygons: list[Polygon]) -> tuple[complex, ...]:
    """"""
    minx = polygons[0][0]
    maxx = minx
    miny = minx
    maxy = miny

    for poly in polygons:
        pminx, pmaxx, pminy, pmaxy = find_minmax(poly)

        if pminx.real < minx.real:
            minx = pminx
        if pmaxx.real > maxx.real:
            maxx = pmaxx
        if pminy.imag < miny.imag:
            miny = pminy
        if pmaxy.imag > maxy.imag:
            maxy = pmaxy

    return minx, maxx, miny, maxy


def create_penrose_rhombus(
    side_length: float, shape: type[RobinsonTriangle] = FatRhombus
) -> RobinsonTriangle:
    """"""
    a = 0j
    b = side_length * np.exp(1j * shape.theta)
    c = 2.0 * b.real + 0j

    # Return a shape centered at the origin
    center = (a + c) / 2.0
    return translate(shape((a, b, c)), -1 * center)


def create_tiling(
    side_length: float,
    image,
    max_n: int = 10,
    initial_shape: type[RobinsonTriangle] = FatRhombus,
):
    """Create a tiling that completely coves the boundaries."""
    tiling = [create_penrose_rhombus(side_length, initial_shape)]

    bounds = Polygon(
        (
            0 + 0j,
            image.shape[1] + 0j,
            image.shape[1] + 1j * image.shape[0],
            0 + 1j * image.shape[0],
        )
    )

    bminx, bmaxx, bminy, bmaxy = find_minmax(bounds)

    x_shift = (bminx + bmaxx) / 2.0
    y_shift = (bminy + bmaxy) / 2.0
    shift = complex(x_shift, y_shift)

    _n = 0
    prev_valid = 0
    while _n < max_n:
        inflated = []
        for t in tiling:
            inflated.extend(inflate(t))
        # tiling = remove_dupes(inflated)
        tiling = inflated
        conj = [conjugate(t) for t in tiling]

        # Translate the tiles so that they're centered within the bounds.
        shift_tiles = [translate(t, shift) for t in remove_dupes(tiling + conj)]

        valid_tiles = list(
            filter(
                lambda t: intersects(t, bounds)
                or is_in_box(t, (bminx.real, bmaxx.real), (bminy.imag, bmaxy.imag)),
                shift_tiles,
            )
        )

        # Converging.
        if len(valid_tiles) <= prev_valid and len(valid_tiles) != 0:
            return valid_tiles

        if prev_valid < len(valid_tiles):
            prev_valid = len(valid_tiles)
        _n += 1

    return valid_tiles


def create_mask(poly: Polygon, image):
    """"""
    pts = np.array([(v.real, v.imag) for v in poly], dtype=np.int32)

    # Create mask
    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.fillPoly(mask, [pts], (255,) * image.shape[2])

    return mask


def make_svg(tiling: ndarray, stroke_width: float = 0.01):
    """Make and return the SVG for the tiling as a str."""

    minx, maxx, miny, maxy = find_minmaxv(tiling)
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
