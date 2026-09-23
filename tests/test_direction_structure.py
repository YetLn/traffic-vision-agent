import unittest

import cv2
import numpy as np
from PIL import Image

from traffic_agent.direction_groups import has_unexplained_structure


def line(text, box, ident):
    return {'text': text, 'bbox': box, 'text_id': ident, 'score': .97}


def multilingual_row():
    rgb = np.full((240, 800, 3), (15, 40, 160), np.uint8)
    cv2.rectangle(rgb, (2, 2), (797, 237), (240, 240, 240), 7)
    # A thin connected supplementary-script row omitted by OCR.
    cv2.line(rgb, (250, 80), (600, 80), (240, 240, 240), 5)
    for x in (280, 390, 540):
        cv2.line(rgb, (x, 80), (x+8, 62), (240, 240, 240), 3)
    texts = [line('和', [240, 110, 300, 170], 't1'),
             line('田', [570, 110, 630, 170], 't2')]
    arrows = [{'arrow_id': 'a1', 'bbox': [80, 105, 180, 175]}]
    return rgb, texts, arrows


class DirectionStructureTests(unittest.TestCase):
    def test_single_character_names_can_anchor_small_supplementary_script(self):
        rgb, texts, arrows = multilingual_row()
        self.assertEqual(has_unexplained_structure(Image.fromarray(rgb), texts, arrows), [])

    def test_independent_large_cross_is_not_supplementary_text(self):
        rgb = np.full((500, 800, 3), (15, 40, 160), np.uint8)
        cv2.line(rgb, (610, 20), (610, 210), (240, 240, 240), 16)
        cv2.line(rgb, (550, 95), (710, 95), (240, 240, 240), 16)
        texts = [line('人民路', [210, 100, 400, 160], 't1')]
        unknown = has_unexplained_structure(Image.fromarray(rgb), texts,
                                           [{'bbox': [80, 100, 150, 165]}])
        self.assertTrue(unknown)
        self.assertTrue(any(row.get('area', 0) > 3000 for row in unknown))

    def test_cross_connected_to_frame_is_not_discarded_with_the_frame(self):
        rgb = np.full((400, 800, 3), (15, 40, 160), np.uint8)
        cv2.rectangle(rgb, (2, 2), (797, 397), (240, 240, 240), 7)
        cv2.line(rgb, (500, 170), (797, 170), (240, 240, 240), 16)
        cv2.line(rgb, (500, 80), (500, 280), (240, 240, 240), 16)
        self.assertTrue(has_unexplained_structure(Image.fromarray(rgb),
                        [line('人民路', [80, 110, 260, 170], 't1')], []))

    def test_shared_two_column_sign_keeps_white_header_as_context(self):
        rgb = np.full((480, 800, 3), (15, 40, 160), np.uint8)
        rgb[:100] = 230
        texts = [line('环城路', [200, 20, 600, 85], 't0'),
                 line('人民路', [40, 150, 280, 210], 't1'),
                 line('黄河路', [510, 150, 750, 210], 't2')]
        self.assertEqual(has_unexplained_structure(Image.fromarray(rgb), texts, []), [])

    def test_large_hollow_route_loop_is_not_removed_as_a_background_hole(self):
        rgb = np.full((400, 800, 3), (15, 40, 160), np.uint8)
        cv2.ellipse(rgb, (550, 200), (105, 130), 0, 0, 360, (240, 240, 240), 16)
        self.assertTrue(has_unexplained_structure(Image.fromarray(rgb),
                        [line('人民路', [80, 110, 260, 170], 't1')], []))

    def test_uncolored_scene_does_not_silently_pass(self):
        self.assertTrue(has_unexplained_structure(Image.new('RGB', (500, 300), 'white'), [], []))


if __name__ == '__main__':
    unittest.main()
