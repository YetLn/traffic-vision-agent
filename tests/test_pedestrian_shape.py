import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from traffic_agent.pedestrian_shape import pedestrian_evidence


def crossing_icon():
    pixels = np.full((160, 160, 3), (30, 80, 175), np.uint8)
    cv2.fillConvexPoly(pixels, np.array([[80, 22], [18, 137], [142, 137]]),
                       (245, 245, 245))
    cv2.circle(pixels, (80, 60), 10, (20, 20, 20), -1)
    cv2.line(pixels, (80, 72), (82, 107), (20, 20, 20), 12)
    cv2.line(pixels, (80, 82), (57, 95), (20, 20, 20), 9)
    cv2.line(pixels, (82, 87), (105, 98), (20, 20, 20), 9)
    cv2.line(pixels, (82, 105), (61, 127), (20, 20, 20), 10)
    cv2.line(pixels, (82, 105), (102, 127), (20, 20, 20), 10)
    for left in (34, 57, 99, 122):
        cv2.line(pixels, (left, 114), (left + 12, 135), (20, 20, 20), 7)
    return Image.fromarray(pixels)


class PedestrianShapeTests(unittest.TestCase):
    def test_triangle_and_pedestrian_pattern_both_required(self):
        crossing = crossing_icon()
        reference = Image.new('RGB', (320, 160), 'white')
        reference.paste(crossing, (0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.png'
            reference.save(path)
            matched = pedestrian_evidence(crossing, path)
            self.assertTrue(matched['accepted'], matched)
            blank_triangle = np.full((160, 160, 3), (30, 80, 175), np.uint8)
            cv2.fillConvexPoly(blank_triangle,
                               np.array([[80, 22], [18, 137], [142, 137]]),
                               (245, 245, 245))
            rejected = pedestrian_evidence(Image.fromarray(blank_triangle), path)
            self.assertFalse(rejected['accepted'])


if __name__ == '__main__':
    unittest.main()
