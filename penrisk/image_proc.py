import cv2
import numpy as np

from .geometry import Polygon


def create_mask(poly: Polygon, image):
    """"""
    pts = np.array([(v.real, v.imag) for v in poly], dtype=np.int32)

    # Create mask
    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.fillPoly(mask, [pts], (255,) * image.shape[2])

    return mask
