import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
from traffic_agent.panel_detector import color_panel_proposals, detect_panels


def result(boxes):
    return {'detections':[{'class_id':2,'class_name':'point-l','bbox':box,'confidence':.8} for box in boxes]}


class PanelDetectorTests(unittest.TestCase):
    def test_padding_runs_real_detector_and_maps_coordinates_to_original(self):
        detector=Mock()
        detector.detect.side_effect=[result([]),result([[600,800,1000,1200]])]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=[]):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        self.assertEqual(detector.detect.call_count,2)
        self.assertEqual(detector.detect.call_args_list[0].args[0].size,(1000,800))
        self.assertEqual(detector.detect.call_args_list[1].args[0].size,(2000,2000))
        self.assertEqual(report['detections'][0]['bbox'],[100,200,500,600])
        self.assertEqual(report['detections'][0]['proposal_source'],'yolo_padded')
        self.assertEqual(report['detections'][0]['confidence'],.8)

    def test_full_panel_deduplicates_truncated_original_box(self):
        detector=Mock()
        detector.detect.side_effect=[result([[100,300,500,600]]),result([[600,800,1000,1200]])]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=[]):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        self.assertEqual(report['total'],1)
        self.assertEqual(report['detections'][0]['bbox'],[100,200,500,600])
        self.assertEqual(len(report['detections'][0]['yolo_support']),2)
        self.assertEqual(len(report['diagnostics']['deduplicated_proposals']),1)

    def test_geometry_without_yolo_support_disabled_by_default(self):
        detector=Mock()
        detector.detect.return_value=result([])
        geometry=[{'bbox':[100,200,500,600],'geometry':{'color':'blue'}}]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=geometry):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
            diagnostic=detect_panels(Image.new('RGB',(1000,800)),detector,allow_geometry_only=True)
        self.assertEqual(report['total'],0)
        self.assertEqual(report['diagnostics']['rejected_proposals'][0]['reason'],'no_yolo_support')
        proposal=diagnostic['detections'][0]
        self.assertFalse(proposal['yolo_supported'])
        self.assertIsNone(proposal['confidence'])
        self.assertEqual(proposal['proposal_source'],'color_geometry')

    def test_geometry_refinement_keeps_real_yolo_seed_and_confidence_kind(self):
        detector=Mock()
        detector.detect.side_effect=[result([[110,300,490,590]]),result([])]
        geometry=[{'bbox':[100,200,500,600],'geometry':{'color':'blue'}}]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=geometry):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        proposal=report['detections'][0]
        self.assertEqual(proposal['seed_yolo_bbox'],[110,300,490,590])
        self.assertEqual(proposal['confidence'],.8)
        self.assertEqual(proposal['proposal_source'],'geometry_with_yolo_support')
        self.assertIn('not_geometry',proposal['confidence_kind'])

    def test_canvas_padding_detection_is_rejected_instead_of_clipped(self):
        detector=Mock()
        detector.detect.side_effect=[result([]),result([[100,100,700,1000]])]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=[]):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        self.assertEqual(report['total'],0)
        self.assertEqual(report['diagnostics']['rejected_proposals'][0]['reason'],'extends_into_padding')

    def test_large_color_region_cannot_expand_valid_sign_into_sky(self):
        detector=Mock()
        detector.detect.side_effect=[result([[300,300,600,500]]),result([])]
        geometry=[{'bbox':[0,0,950,700],'geometry':{'color':'blue'}},
                  {'bbox':[10,10,950,700],'geometry':{'color':'blue'}}]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=geometry):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        self.assertEqual(report['total'],1)
        self.assertEqual(report['detections'][0]['bbox'],[300,300,600,500])
        self.assertEqual(report['diagnostics']['rejected_proposals'][0]['reason'],'touches_image_edge')

    def test_exif_is_normalized_before_both_passes(self):
        detector=Mock();detector.detect.return_value=result([])
        image=Image.new('RGB',(1000,800));image.getexif()[274]=6
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=[]):
            report=detect_panels(image,detector)
        self.assertEqual(report['image_size'],[800,1000])
        self.assertEqual(detector.detect.call_args_list[0].args[0].size,(800,1000))

    def test_color_geometry_finds_blue_panel_with_white_header(self):
        image=Image.new('RGB',(600,500),'gray')
        draw=ImageDraw.Draw(image)
        draw.rectangle((100,80,500,420),fill='#1248b0')
        draw.rectangle((108,88,492,170),fill='white')
        draw.rectangle((170,220,420,250),fill='white')
        proposals=color_panel_proposals(image)
        self.assertTrue(proposals)
        self.assertLessEqual(proposals[0]['bbox'][1],80)
        self.assertGreaterEqual(proposals[0]['bbox'][3],420)

    def test_solid_blue_field_is_not_a_panel_proposal(self):
        image=Image.new('RGB',(600,500),'gray')
        ImageDraw.Draw(image).rectangle((100,80,500,420),fill='#1248b0')
        self.assertEqual(color_panel_proposals(image),[])

    def test_three_border_panel_requires_a_real_yolo_seed(self):
        detector=Mock()
        detector.detect.side_effect=[result([[200,150,600,400]]),result([])]
        quad=[[120,220],[650,120],[650,600],[120,600]]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=[]), \
             patch('traffic_agent.panel_detector.estimate_supported_quad',return_value=(quad,{
                 'edge_support':[.98,.33,.96,.97]})):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        self.assertEqual(report['total'],1)
        panel=report['detections'][0]
        self.assertEqual(panel['proposal_source'],'geometry_with_yolo_support')
        self.assertTrue(panel['yolo_supported'])
        self.assertEqual(panel['seed_yolo_bbox'],[200,150,600,400])
        self.assertEqual(panel['geometry']['source_quad'],quad)

    def test_near_frontal_extra_quad_does_not_expand_seed_crop(self):
        detector=Mock()
        detector.detect.side_effect=[result([[200,150,600,400]]),result([])]
        quad=[[120,120],[650,120],[650,600],[120,600]]
        with patch('traffic_agent.panel_detector.color_panel_proposals',return_value=[]), \
             patch('traffic_agent.panel_detector.estimate_supported_quad',return_value=(quad,{
                 'edge_support':[1.,1.,1.,1.]})):
            report=detect_panels(Image.new('RGB',(1000,800)),detector)
        self.assertEqual(report['detections'][0]['bbox'],[200,150,600,400])
        self.assertEqual(report['detections'][0]['proposal_source'],'yolo_original')


if __name__=='__main__':
    unittest.main()
