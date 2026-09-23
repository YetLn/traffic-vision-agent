import unittest

import cv2
import numpy as np
from PIL import Image

from traffic_agent.direction_geometry import classify_arrow_mask, detect_direction_arrows


def arrow_mask():
    """A hand-specified held-out polygon, independent of the template generator."""
    mask = np.zeros((160, 105), np.uint8)
    points = np.array([[52, 4], [99, 63], [68, 47], [68, 151],
                       [36, 151], [36, 47], [5, 63]])
    cv2.fillPoly(mask, [points], 1)
    return mask


class DirectionGeometryTests(unittest.TestCase):
    def test_cardinal_rotations_use_shape(self):
        for turn, expected in enumerate(("up", "left", "down", "right")):
            with self.subTest(expected=expected):
                result = classify_arrow_mask(np.rot90(arrow_mask(), turn))
                self.assertEqual(result["direction"], expected, result)

    def test_negative_shapes_abstain(self):
        for name in ("rectangle", "circle", "cross", "triangle", "bent", "hollow", "double_head"):
            shape = np.zeros((170, 120), np.uint8)
            if name == "rectangle":
                cv2.rectangle(shape, (35, 5), (80, 160), 1, -1)
            elif name == "circle":
                cv2.ellipse(shape, (60, 85), (35, 65), 0, 0, 360, 1, -1)
            elif name == "cross":
                cv2.rectangle(shape, (45, 5), (75, 160), 1, -1)
                cv2.rectangle(shape, (10, 70), (110, 100), 1, -1)
            elif name == "triangle":
                cv2.fillPoly(shape, [np.array([[60, 5], [10, 160], [110, 160]])], 1)
            elif name == "bent":
                cv2.fillPoly(shape, [np.array([[5, 50], [42, 15], [42, 35], [90, 35],
                                              [90, 155], [65, 155], [65, 65], [42, 65], [42, 85]])], 1)
            elif name == "hollow":
                cv2.rectangle(shape, (15, 10), (100, 160), 1, 8)
            else:
                cv2.fillPoly(shape, [np.array([[60, 5], [110, 50], [75, 50], [75, 120],
                                              [110, 120], [60, 165], [10, 120], [45, 120],
                                              [45, 50], [10, 50]])], 1)
            with self.subTest(shape=name):
                self.assertIsNone(classify_arrow_mask(shape)["direction"])

    def test_short_wide_arrow_and_diagonal(self):
        wide = cv2.resize(arrow_mask(), (135, 110), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(classify_arrow_mask(wide)["direction"], "up")
        diagonal = cv2.warpAffine(arrow_mask(),
                                 cv2.getRotationMatrix2D((80, 80), 45, 1), (210, 210))
        self.assertIsNone(classify_arrow_mask(diagonal)["direction"])

    def test_blanks_and_multiple_components_abstain(self):
        self.assertIsNone(classify_arrow_mask(np.zeros((20, 20)))["direction"])
        shape = arrow_mask()
        shape[155:159, :4] = 1
        self.assertEqual(classify_arrow_mask(shape)["reason"], "multiple_components")

    def test_image_detector_dark_blue_and_text_mask(self):
        rgb = np.empty((400, 500, 3), np.uint8)
        rgb[:] = [8, 22, 100]
        mask = arrow_mask()
        roi = rgb[120:280, 50:155]
        roi[mask > 0] = [95, 110, 175]
        detected = detect_direction_arrows(Image.fromarray(rgb))
        self.assertEqual([row["direction"] for row in detected["arrows"]], ["up"])
        self.assertEqual(detected["arrows"][0]["bbox"], [55, 124, 150, 272])
        # Text lies on the right; it must never flip this up arrow into a left.
        with_text = detect_direction_arrows(Image.fromarray(rgb), [[250, 120, 420, 220]])
        self.assertEqual(with_text["arrows"][0]["direction"], "up")
        masked = detect_direction_arrows(Image.fromarray(rgb), [[40, 110, 160, 290]])
        self.assertEqual(masked["status"], "abstain")

    def test_component_crossing_crop_edge_is_rejected(self):
        rgb = np.full((160, 250, 3), 20, np.uint8)
        mask = arrow_mask()[4:]
        rgb[:156, :105][mask > 0] = 230
        self.assertEqual(detect_direction_arrows(rgb)["status"], "abstain")

    def test_dimmer_connected_branch_is_not_cut_into_arrow(self):
        rgb = np.full((340, 400, 3), 20, np.uint8)
        # A separate bright panel creates thresholds above the dim branch.
        cv2.rectangle(rgb, (275, 0), (399, 330), (180, 180, 180), -1)
        cv2.line(rgb, (102, 210), (245, 210), (140, 140, 140), 12)
        rgb[70:230, 50:155][arrow_mask() > 0] = 235
        self.assertEqual(detect_direction_arrows(rgb)["status"], "abstain")

    def test_small_foreground_overlap_does_not_erase_whole_arrow(self):
        rgb = np.full((300, 450, 3), (8, 22, 100), np.uint8)
        rgb[60:220, 50:155][arrow_mask() > 0] = [160, 180, 220]
        # This OCR box grazes the broad head's right edge; its blank margin
        # covers far more of the arrow's bbox than actual arrow pixels.
        result = detect_direction_arrows(rgb, [[135, 50, 400, 170]])
        self.assertEqual([row['direction'] for row in result['arrows']], ['up'])
        arrow = result['arrows'][0]
        self.assertEqual(arrow['bbox'], [55, 64, 150, 212])
        self.assertGreater(arrow['foreground_text_fraction'], 0)
        self.assertLess(arrow['foreground_text_fraction'], 0.20)
        self.assertGreater(arrow['bbox_text_fraction'], arrow['foreground_text_fraction'])
        self.assertEqual(result['text_exclusion']['retained_despite_bbox_intersection'], 1)
        duplicate = detect_direction_arrows(rgb, [[135, 50, 400, 170]] * 8)
        self.assertEqual(duplicate['arrows'][0]['foreground_text_fraction'],
                         arrow['foreground_text_fraction'])

    def test_enclosed_arrow_like_text_is_still_rejected(self):
        rgb = np.full((300, 450, 3), 20, np.uint8)
        rgb[60:220, 50:155][arrow_mask() > 0] = 230
        # Even a perfectly arrow-shaped glyph inside an OCR line is not
        # evidence of an independent arrow; duplicated boxes do not alter it.
        result = detect_direction_arrows(rgb, [[45, 55, 160, 225]] * 2)
        self.assertEqual(result['status'], 'abstain')
        self.assertGreater(result['rejected'].get('overlaps_text', 0), 0)

    def test_substantial_partial_text_coverage_still_rejects(self):
        rgb = np.full((300, 450, 3), 20, np.uint8)
        rgb[60:220, 50:155][arrow_mask() > 0] = 230
        # A text box covering the central shaft, though not the arrow bbox,
        # carries substantial foreground evidence and must remain a conflict.
        result = detect_direction_arrows(rgb, [[85, 115, 125, 215]])
        self.assertEqual(result['status'], 'abstain')
        self.assertGreater(result['rejected'].get('overlaps_text', 0), 0)

    def test_ocr_overlap_never_cuts_a_connected_diagram(self):
        rgb = np.full((340, 400, 3), 20, np.uint8)
        cv2.line(rgb, (102, 210), (245, 210), (230, 230, 230), 12)
        rgb[70:230, 50:155][arrow_mask() > 0] = 230
        # An OCR box covering only the branch must not erase that branch and
        # manufacture a freestanding up arrow from the remaining component.
        result = detect_direction_arrows(rgb, [[157, 197, 252, 224]])
        self.assertEqual(result['status'], 'abstain')


if __name__ == "__main__":
    unittest.main()
