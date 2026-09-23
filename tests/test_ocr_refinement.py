from copy import deepcopy
import unittest
from unittest.mock import patch

from PIL import Image

from traffic_agent.ocr_refinement import refine_ocr_lines


def observation(text='北京路', bbox=None, score=.97):
    x, y, r, b = bbox or [30, 60, 250, 110]
    return {'text_id': 't1', 'text': text, 'score': score,
            'bbox': [x, y, r, b], 'quad': [[x, y], [r, y], [r, b], [x, b]],
            'center': [(x+r)/2, (y+b)/2], 'height': b-y}


def candidate(box=None):
    return {'arrow_id': 'a1', 'direction': 'right', 'bbox': box or [220, 55, 300, 120],
            'shape_iou': .94, 'direction_margin': .3}


class OcrRefinementTests(unittest.TestCase):
    def run_refinement(self, text_lines=None, rechecked=None, arrows=None):
        text_lines = [observation()] if text_lines is None else text_lines
        rechecked = [observation(bbox=[10, 14, 178, 63], score=.99)] if rechecked is None else rechecked
        arrows = [candidate()] if arrows is None else arrows
        with patch('traffic_agent.ocr_refinement.detect_direction_arrows',
                   return_value={'method': 'test_intact_shapes', 'arrows': arrows}), \
             patch('traffic_agent.ocr_refinement.read_text', return_value={'lines': rechecked}) as ocr:
            result = refine_ocr_lines(Image.new('RGB', (350, 200), 'blue'), text_lines)
        return result, ocr

    def test_exact_recheck_refines_coordinates_and_preserves_original(self):
        supplied = [observation()]
        before = deepcopy(supplied)
        result, ocr = self.run_refinement(supplied)
        self.assertEqual(supplied, before)
        line = result['lines'][0]
        self.assertEqual(line['text'], before[0]['text'])
        self.assertEqual(line['score'], .97)
        self.assertEqual(line['bbox'], [32.0, 61.0, 200.0, 110.0])
        self.assertEqual(line['ocr_refinement']['original'], before[0])
        self.assertEqual(line['ocr_refinement']['rechecked']['segments'][0]['center'], [116.0, 85.5])
        self.assertEqual(result['diagnostics']['modified_count'], 1)
        self.assertEqual(ocr.call_count, 1)

    def test_ordered_fragments_must_reproduce_all_original_characters(self):
        result, _ = self.run_refinement(rechecked=[observation('路', [109, 14, 178, 63]),
                                                  observation('北京', [10, 14, 108, 63])])
        self.assertEqual(result['diagnostics']['modified_count'], 1)
        self.assertEqual(result['lines'][0]['bbox'], [32.0, 61.0, 200.0, 110.0])
        self.assertEqual(len(result['lines'][0]['ocr_refinement']['rechecked']['segments']), 2)

    def test_left_edge_candidate_keeps_coordinates_in_input_crop(self):
        supplied = [observation(bbox=[100, 60, 310, 110])]
        result, _ = self.run_refinement(supplied,
            rechecked=[observation(bbox=[10, 14, 180, 63])], arrows=[candidate([50, 55, 125, 120])])
        self.assertEqual(result['lines'][0]['bbox'], [137.0, 61.0, 307.0, 110.0])
        self.assertEqual(result['lines'][0]['ocr_refinement']['coordinate_frame'], 'input_crop_pixels')

    def test_changed_missing_and_duplicate_text_all_retain_original_box(self):
        for rechecked in ([observation('京路', [10, 14, 178, 63])],
                          [observation('北京路路', [10, 14, 178, 63])], []):
            with self.subTest(rechecked=rechecked):
                result, _ = self.run_refinement(rechecked=rechecked)
                self.assertEqual(result['lines'], [observation()])
                self.assertEqual(result['diagnostics']['modified_count'], 0)

    def test_low_confidence_or_roi_edge_does_not_authorize_trimming(self):
        for row in (observation(bbox=[10, 14, 178, 63], score=.8),
                    observation(bbox=[0, 14, 178, 63])):
            result, _ = self.run_refinement(rechecked=[row])
            self.assertEqual(result['lines'], [observation()])

    def test_arrow_shaped_glyph_inside_text_is_not_a_trim_proposal(self):
        result, ocr = self.run_refinement(arrows=[candidate([80, 55, 150, 120])])
        self.assertEqual(result['lines'], [observation()])
        ocr.assert_not_called()

    def test_multiple_edge_candidates_keep_ambiguous_span(self):
        result, ocr = self.run_refinement(arrows=[candidate(), candidate([10, 55, 45, 120])])
        self.assertEqual(result['diagnostics']['attempts'][0]['reason'], 'multiple_edge_candidates')
        self.assertEqual(result['lines'], [observation()])
        ocr.assert_not_called()

    def test_no_intact_arrow_does_not_modify_a_connected_map(self):
        result, ocr = self.run_refinement(arrows=[])
        self.assertEqual(result['lines'], [observation()])
        ocr.assert_not_called()

    def test_failed_local_ocr_leaves_input_untouched(self):
        supplied = [observation()]
        with patch('traffic_agent.ocr_refinement.detect_direction_arrows', return_value={'arrows': [candidate()]}), \
             patch('traffic_agent.ocr_refinement.read_text', side_effect=RuntimeError('unavailable')):
            result = refine_ocr_lines(Image.new('RGB', (350, 200)), supplied)
        self.assertEqual(result['lines'], supplied)
        self.assertEqual(result['diagnostics']['attempts'][0]['reason'], 'ocr_recheck_unavailable')


if __name__ == '__main__':
    unittest.main()
