import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from traffic_agent.small_sign_reader import _features, _number, _numeric_override, match_crop


def blue_arrow(direction):
    image = np.full((128, 128, 3), (20, 60, 190), np.uint8)
    if direction == 'right':
        cv2.rectangle(image, (37, 51), (78, 75), (255, 255, 255), -1)
        cv2.fillPoly(image, [np.array([[72, 28], [108, 64], [72, 100]])], (255, 255, 255))
    else:
        cv2.rectangle(image, (50, 51), (91, 75), (255, 255, 255), -1)
        cv2.fillPoly(image, [np.array([[56, 28], [20, 64], [56, 100]])], (255, 255, 255))
    return Image.fromarray(image)


class SmallSignReaderTests(unittest.TestCase):
    def setUp(self):
        right = blue_arrow('right')
        left = blue_arrow('left')
        self.right = right
        self.refs = [
            ({'id': 'r', 'name': '向右转弯标志', 'dataset_class_id': 3,
              'meaning_summary': '按标志要求右转。', 'source_image_url': 'https://example.test/r'},
             _features(np.asarray(right))),
            ({'id': 'l', 'name': '向左转弯标志', 'dataset_class_id': 3,
              'meaning_summary': '按标志要求左转。', 'source_image_url': 'https://example.test/l'},
             _features(np.asarray(left))),
        ]

    def test_clear_distinct_icon_can_be_explained(self):
        with patch('traffic_agent.small_sign_reader._ocr_lines', return_value=[]):
            result = match_crop(self.right, 3, self.refs)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['name'], '向右转弯标志')
        self.assertEqual(result['catalog_id'], 'r')

    def test_small_and_wrong_color_abstain(self):
        self.assertIn('小于', match_crop(self.right.resize((50, 50)), 3, self.refs)['reason'])
        red = Image.new('RGB', (128, 128), (190, 30, 30))
        self.assertFalse(match_crop(red, 3, self.refs)['accepted'])

    def test_equal_candidates_abstain(self):
        duplicates = [(dict(entry, name='same-' + str(i)), feature)
                      for i, (entry, feature) in enumerate(self.refs[:1] * 2)]
        result = match_crop(self.right, 3, duplicates)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['margin'], 0)

    def test_pedestrian_rule_requires_reviewed_catalog_entry(self):
        crossing = {'id': 'point-s-017', 'name': '人行横道标志',
                    'dataset_class_id': 3, 'meaning_summary': '指明该处有人行横道。',
                    'source_image_url': 'https://example.test/crossing'}
        refs = self.refs + [(crossing, _features(np.asarray(self.right)))]
        with patch('traffic_agent.small_sign_reader.pedestrian_evidence',
                   return_value={'accepted': True, 'pattern_similarity': 0.8}) as check:
            result = match_crop(self.right, 3, refs)
        self.assertTrue(check.called)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['catalog_id'], 'point-s-017')
        self.assertEqual(result['meaning'], crossing['meaning_summary'])
        self.assertEqual(result['pattern_similarity'], 0.8)

    def test_roundabout_is_not_claimed_from_generic_similarity(self):
        roundabout = {'id': 'point-s-010', 'name': '环岛行驶标志',
                      'dataset_class_id': 3, 'meaning_summary': '车辆按图示方向环行。',
                      'source_image_url': 'https://example.test/roundabout'}
        refs = [(roundabout, _features(np.asarray(self.right))), self.refs[1]]
        result = match_crop(self.right, 3, refs)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['candidate'], '环岛行驶标志')
        self.assertIn('整体图案相似度', result['reason'])

    def test_number_requires_ocr_and_matching_ring(self):
        image = np.full((128, 128, 3), 255, np.uint8)
        cv2.circle(image, (64, 64), 53, (220, 20, 20), 14)
        cv2.putText(image, '40', (33, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 0), 4, cv2.LINE_AA)
        crop = Image.fromarray(image)
        refs = [({'id': 'speed', 'name': '限制速度', 'dataset_class_id': 1,
                  'meaning_summary': '不得超过牌面数值。',
                  'source_image_url': 'https://example.test/speed'},
                 _features(image))]
        line = {'text': '40', 'score': 0.99, 'center': [64, 64]}
        with patch('traffic_agent.small_sign_reader._ocr_lines', return_value=[line]):
            result = match_crop(crop, 1, refs)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['number']['value'], 40)

    def test_speed_number_uses_sharpened_ocr_only_after_raw_miss(self):
        crop = Image.new('RGB', (128, 128), 'white')
        with patch('traffic_agent.small_sign_reader.read_text', side_effect=[
            {'lines': []}, {'lines': [{'text': '60', 'score': 0.99}]}
        ]) as ocr:
            number = _number(crop, '限制速度')
        self.assertEqual(number['value'], 60)
        self.assertEqual(ocr.call_count, 2)
        self.assertIsNot(ocr.call_args_list[1].args[0], crop)

    def test_speed_override_rejects_off_center_or_unit_bearing_text(self):
        image = np.full((128, 128, 3), 255, np.uint8)
        cv2.circle(image, (64, 64), 53, (220, 20, 20), 14)
        crop = Image.fromarray(image)
        refs = [({'id': 'speed', 'name': '限制速度', 'dataset_class_id': 1,
                  'meaning_summary': '不得超过牌面数值。',
                  'source_image_url': 'https://example.test/speed'},
                 _features(image))]
        for text, center in (('40', [64, 53]), ('4.5m', [64, 64])):
            with self.subTest(text=text, center=center), patch(
                'traffic_agent.small_sign_reader._ocr_lines',
                return_value=[{'text': text, 'score': 0.99, 'center': center}]
            ):
                self.assertIsNone(_numeric_override(crop, image, refs, 1))


if __name__ == '__main__':
    unittest.main()
