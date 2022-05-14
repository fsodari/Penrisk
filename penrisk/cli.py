import argparse
import sys

"""
There are several independent programs that you can run from the command line.

Penrose Tile Generator.



Image Slicer.

This program will slice up an image into individual images for each tile.

    --image image.png           # Image to be sliced. If no image is given, masks can still be generated.
    --tiling tiles.svg/.npy     # The list of tiles to use for slicing.
    --mask prefix               # Generate the mask files. Saved as prefixN.png
    --prefix output             # Output file prefix. Output images are saved as prefixN.png
    --crop                      # Crop the output images to the bounding box of the tile.
    --rotate                    # Rotate each tile so that its symmetric about the y axis and the x axis.
    --normalize                 # Perform both the rotation and the crop.

Step Slicer.

This program will generate 3D tiles from an image/countour map. The value of the pixels represent the height above the base.

"""


def tiler():
    """
    This program will generate a tile pattern and save it as an .svg file.
    The pattern will cover a rectangle with the dimensions "shape". Some tiles may stick out
    beyond the border. The program will iterate until it can cover the rectangle.

    --image some_file.svg       # Output file name.
    --tiling tile_data.npy      # Save the tiling array as a numpy save file.
    --tile-size size 1.0        # Side length of the rhombuses
    --shape 1080 600            # Output image dimensions. If none given, it will run for max_n iterations.
    --max_n 20                  # Maximum number of iterations. It can take long time to inflate.
    --stroke_width 0.1          # Width of the path in the svg image.
    --translate 100 -42         # Translate the tiling to change the pattern that shows in the svg.
    """

    parser = argparse.ArgumentParser(description="Penrose Tile Generator.")
    parser.add_argument(
        "--image",
        metavar="image.png",
        help="output file name",
    )
    parser.add_argument(
        "--tiles",
        metavar="tiles.npy",
        help="Save numpy tile data.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        help="Side length of the tiles",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        help="Output image dimensions",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of iterations before quitting.",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=0.1,
        help="Stroke width in svg file.",
    )
    parser.add_argument(
        "--translate",
        type=float,
        nargs="+",
        default=(0, 0),
        help="Translate tiles before saving in svg.",
    )

    # If no arguments given, print help.
    if len(sys.argv) < 2:
        parser.print_usage()

    args = parser.parse_args()

    print(args)


if __name__ == "__main__":
    tiler()
