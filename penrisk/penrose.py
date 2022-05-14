from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy import ndarray

from .geometry import Polygon, intersects, isclose, PSI, PSI2, translate
from .geometry import centroid, conjugate, rotate, scale

from tempfile import NamedTemporaryFile
from cairosvg import svg2png

import cv2


class RobinsonTriangle(Polygon):
    id: type[RobinsonTriangle]
    theta: float

    def __new__(cls, input_array: ndarray, id: type[RobinsonTriangle] = None):
        a, b, c, *_ = input_array

        # Check if ab == bc
        if not isclose(abs(b - a), abs(c - b)):
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

    #   This seems rather difficult to do.
    #   Checking if each element is equal to every other element is kind of slow O(n**2)
    #   If we sort the list, it should require fewer comparisons since we only have to check
    #   a certain range.
    # Sort elements.
    selements = sorted(tiles, key=lambda e: (centroid(e).real, centroid(e).imag))

    # First item is unique.
    unique_elements = [selements[0]]

    # Go through the entire list
    for element in selements:
        # Check if the element is already in the unique elements list.
        found = False
        for unique in unique_elements:
            """"""
            # Sorted by real first. If we move past an element, we can skip early.
            if isclose(centroid(unique), centroid(element)):
                found = True
                break

        if not found:
            unique_elements.append(element)

    return unique_elements


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
    shape: tuple[int, ...],
    n: int = 10,
    initial_shape: type[RobinsonTriangle] = FatRhombus,
    offset: complex = 0j,
):
    """Create a tiling that completely coves the boundaries."""
    tiling = [create_penrose_rhombus(side_length, initial_shape)]

    # Since we'll be reflecting the entire tileset across the x axis,
    # we just need to check half of the image.
    bounds = Polygon(
        (
            -shape[1] / 2 + 0j,
            shape[1] / 2 + 0j,
            shape[1] / 2 + 1j * shape[0] / 2,
            -shape[1] / 2 + 1j * shape[0] / 2,
        )
    )

    l1 = Polygon(np.array([bounds[0], bounds[3]]))

    for _ in range(n):
        # inflated
        inflated: list[RobinsonTriangle] = []
        for t in tiling:
            inflated.extend(inflate(t))
        tiling = inflated

        # Tiling bounds.
        minx, _, _, maxy = find_minmaxv(tiling)

        l2 = Polygon(np.array([minx, maxy]))
        if minx.real < -shape[1] / 2 and not intersects(l1, l2):
            break

    tiling = remove_dupes(tiling + [conjugate(t) for t in tiling])

    full_bounds = Polygon(
        (
            -shape[1] / 2 - 1j * shape[0] / 2,
            shape[1] / 2 - 1j * shape[0] / 2,
            shape[1] / 2 + 1j * shape[0] / 2,
            -shape[1] / 2 + 1j * shape[0] / 2,
        )
    )

    valid_tiles = list(
        filter(
            lambda t: intersects(t, full_bounds)
            or is_in_box(
                t, (-shape[1] / 2, shape[1] / 2), (-shape[0] / 2, shape[0] / 2)
            ),
            tiling,
        )
    )

    # Rotating will make it fit better on rectangular images.
    # tiling = [rotate(t, PSI) for t in tiling]
    shift_amount = complex(shape[1] / 2, shape[0] / 2)
    return [translate(v, shift_amount) for v in valid_tiles]


def find_largest_rectangle(tiling: list):
    """
    Find the largest rectangle that can fit entirely contained in
    a set of tiles.

    The tiling is rhombus shaped.
    """


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


def overlay_tiles(
    tiling: ndarray, im: cv2.Mat, output_image: Path, stroke_width: float = 0.01
):
    """Draw tiles on top of an image."""
    # Draw each polygon
    for tile in tiling:
        # List of points in the polygon. Convert complex to cartesian.
        pts = np.array([[x.real, x.imag] for x in tile], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(im, [pts], True, (0, 0, 0), stroke_width, lineType=cv2.LINE_AA)

    cv2.imwrite(output_image, im)


if __name__ == "__main__":
    pts = 1j, 0, 1.0
    x = rotate(pts, np.pi / 10.0)
    tri = RobinsonTriangle(x)
