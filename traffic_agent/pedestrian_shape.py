"""Conservative structure check for a blue pedestrian-crossing sign.

The white triangle and dark pedestrian/crosswalk pattern must both agree with
the reviewed official illustration. Scores are image similarities, not odds.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


TRIANGLE = np.zeros((128, 128), np.uint8)
cv2.fillConvexPoly(TRIANGLE, np.array([[64, 18], [13, 109], [115, 109]],
                                      np.int32), 1)
INNER = np.zeros((128, 128), bool)
INNER[12:116, 12:116] = True
OUTSIDE = INNER & (TRIANGLE == 0)


def _masks(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = (((hsv[:, :, 0] >= 88) & (hsv[:, :, 0] <= 140) &
             (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 45)).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    sign = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(sign)
    if min(width, height) < 55 or not 0.68 <= width / height <= 1.5:
        return None
    region = np.zeros(blue.shape, np.uint8)
    cv2.drawContours(region, [cv2.convexHull(sign)], -1, 255, -1)
    rgb = cv2.resize(rgb[y:y+height, x:x+width], (128, 128))
    region = cv2.resize(region[y:y+height, x:x+width], (128, 128))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = (hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 145) & (region > 0)
    dark = (hsv[:, :, 2] < 115) & (hsv[:, :, 1] < 100) & (region > 0)
    return white, dark & (TRIANGLE > 0)


@lru_cache(maxsize=1)
def _template(reference_path):
    if not Path(reference_path).is_file():
        return None
    with Image.open(reference_path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    left_variant = rgb[:, :rgb.shape[1] // 2]
    masks = _masks(left_variant)
    if masks is None:
        return None
    dark = masks[1]
    return tuple(np.roll(np.roll(dark, dy, axis=0), dx, axis=1)
                 for dy in (-8, -4, 0, 4, 8)
                 for dx in (-8, -4, 0, 4, 8))


def pedestrian_evidence(crop, reference_path):
    masks = _masks(np.asarray(ImageOps.exif_transpose(crop).convert('RGB')))
    templates = _template(str(reference_path))
    if masks is None or templates is None:
        return {'accepted': False, 'reason': '蓝底牌面或官方参照图不可用'}
    white, dark = masks
    inside = float((white & (TRIANGLE > 0)).sum() / TRIANGLE.sum())
    outside = float((white & OUTSIDE).sum() / OUTSIDE.sum())
    ink = float(dark.sum() / TRIANGLE.sum())
    dark_count = int(dark.sum())
    dice = max(float(2 * (dark & template).sum() /
                     max(1, dark_count + int(template.sum())))
               for template in templates)
    return {'accepted': inside >= .24 and outside <= .10 and
            ink >= .16 and dice >= .45,
            'white_triangle_share': round(inside, 3),
            'white_outside_share': round(outside, 3),
            'dark_pattern_share': round(ink, 3),
            'pattern_similarity': round(dice, 3)}
