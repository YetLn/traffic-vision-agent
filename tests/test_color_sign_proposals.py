import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from traffic_agent.color_sign_proposals import color_sign_proposals
from traffic_agent.small_sign_reader import read_small_signs


class FakeDetector:
    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect(self, image, conf):
        return {'detections': self.detections}


class ColorSignProposalTests(unittest.TestCase):
    def test_red_circle_proposed_in_original_coordinates(self):
        pixels = np.full((320, 400, 3), 255, np.uint8)
        cv2.circle(pixels, (230, 130), 55, (220, 20, 20), 13)
        proposals = color_sign_proposals(Image.fromarray(pixels))
        self.assertTrue(any(p['class_id'] == 1 and
                            p['bbox'][0] <= 175 and p['bbox'][2] >= 285 and
                            p['bbox'][1] <= 75 and p['bbox'][3] >= 185
                            for p in proposals))

    def test_only_verified_new_proposal_is_shown(self):
        image = Image.new('RGB', (300, 200), 'white')
        proposals = [{'class_id': 1, 'bbox': [20, 20, 130, 130],
                      'color_fraction': 0.3},
                     {'class_id': 3, 'bbox': [150, 20, 260, 130],
                      'color_fraction': 0.4}]
        with patch('traffic_agent.small_sign_reader._references', return_value=(
            {'source_page': 'https://example.test'}, [('entry', 'feature')]
        )), patch('traffic_agent.small_sign_reader.color_sign_proposals',
                  return_value=proposals), patch(
            'traffic_agent.small_sign_reader.match_crop', side_effect=[
                {'accepted': True, 'name': '限制速度'},
                {'accepted': False, 'reason': 'not verified'}
            ]):
            report = read_small_signs(image, FakeDetector())
        self.assertEqual(report['color_proposals'], 2)
        self.assertEqual(report['color_proposals_accepted'], 1)
        self.assertEqual(len(report['signs']), 1)
        self.assertEqual(report['signs'][0]['detector_source'], 'color_shape')
        self.assertIsNone(report['signs'][0]['detection_confidence'])

    def test_overlap_with_yolo_box_is_not_duplicated(self):
        image = Image.new('RGB', (200, 200), 'white')
        proposal = {'class_id': 1, 'bbox': [40, 40, 150, 150],
                    'color_fraction': 0.3}
        yolo = {'class_id': 1, 'class_name': 'ban', 'confidence': 0.9,
                'bbox': [40, 40, 150, 150]}
        with patch('traffic_agent.small_sign_reader._references', return_value=(
            {'source_page': 'https://example.test'}, [('entry', 'feature')]
        )), patch('traffic_agent.small_sign_reader.color_sign_proposals',
                  return_value=[proposal]), patch(
            'traffic_agent.small_sign_reader.match_crop',
            return_value={'accepted': True, 'name': '限制速度'}
        ) as matcher:
            report = read_small_signs(image, FakeDetector([yolo]))
        self.assertEqual(len(report['signs']), 1)
        self.assertEqual(report['color_proposals_accepted'], 0)
        self.assertEqual(matcher.call_count, 1)


if __name__ == '__main__':
    unittest.main()
