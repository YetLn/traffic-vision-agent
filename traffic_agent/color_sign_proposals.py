"""Experimental color/shape sign proposals in original image coordinates.

These are candidate boxes, not calibrated YOLO detections or classifications.
The small-sign reader must still verify each proposal before explaining it.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def color_sign_proposals(image: Image.Image, min_side: int = 80):
    rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    masks = {
        0: (h >= 15) & (h <= 40) & (s >= 55) & (v >= 55),
        1: ((h <= 12) | (h >= 170)) & (s >= 60) & (v >= 50),
        3: (h >= 88) & (h <= 140) & (s >= 55) & (v >= 45),
    }
    kernel = np.ones((3, 3), np.uint8)
    proposals = []
    for class_id, pixels in masks.items():
        mask = cv2.morphologyEx((pixels.astype(np.uint8) * 255),
                                cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, ht = cv2.boundingRect(contour)
            if min(w, ht) < min_side or not 0.68 <= w / ht <= 1.45:
                continue
            color_fraction = float(np.count_nonzero(mask[y:y+ht, x:x+w])) / (w * ht)
            if color_fraction < {0: .12, 1: .06, 3: .16}[class_id]:
                continue
            pad = round(.06 * max(w, ht))
            box = [max(0, x-pad), max(0, y-pad),
                   min(width, x+w+pad), min(height, y+ht+pad)]
            proposals.append({'class_id': class_id, 'bbox': box,
                              'color_fraction': round(color_fraction, 3),
                              'source': 'color_shape'})
    proposals.sort(key=lambda row: -((row['bbox'][2]-row['bbox'][0]) *
                                     (row['bbox'][3]-row['bbox'][1])))
    return proposals[:30]
