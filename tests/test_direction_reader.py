import unittest
from unittest.mock import patch, Mock
from PIL import Image

from traffic_agent.direction_reader import associate_rows, read_directions, _map_evidence
from traffic_agent.ocr import read_text


def line(text='人民路', box=None, score=0.96):
    box = box or [120, 100, 260, 140]
    x,y,r,b = box
    return {'text_id':'t1', 'text':text, 'score':score, 'bbox':box,
            'quad':[[x,y],[r,y],[r,b],[x,b]], 'center':[(x+r)/2,(y+b)/2]}


def arrow(box=None, direction='up'):
    return {'arrow_id':'a1', 'bbox':box or [40,90,90,150], 'direction':direction,
            'center':[65,120], 'shape_iou':0.92, 'direction_margin':0.31}


class DirectionReaderTests(unittest.TestCase):
    def test_direction_comes_from_arrow_not_relative_position(self):
        for direction in ('up','left','right'):
            relations, reasons = associate_rows([line()], [arrow(direction=direction)], (400,300))
            self.assertEqual(relations[0]['direction'], direction)
            self.assertFalse(reasons)

    def test_rejects_missing_arrow_and_low_text_and_prompt(self):
        self.assertEqual(associate_rows([line()], [], (400,300))[1], ['no_arrow'])
        self.assertEqual(associate_rows([line(score=.6)], [arrow()], (400,300))[1], ['low_text_confidence'])
        self.assertEqual(associate_rows([line('请按导向车道行驶')], [arrow()], (400,300))[1], ['prompt_sign'])

    def test_two_labels_sharing_one_arrow_abstain(self):
        extra = line('黄河路', [120,135,260,175]); extra['text_id']='t2'
        self.assertEqual(associate_rows([line(),extra], [arrow()], (400,300))[0], [])

    def test_label_at_crop_boundary_abstains_instead_of_inventing_full_name(self):
        for edge in (400,397):
            clipped=line('人民', [120,100,edge,140])
            self.assertEqual(associate_rows([clipped], [arrow()], (400,300))[1], ['clipped_text'])

    def test_ambiguous_two_arrows_same_row_abstain(self):
        a2 = arrow([50,90,100,150]); a2['arrow_id']='a2'
        self.assertEqual(associate_rows([line()], [arrow(),a2], (400,300))[1], ['ambiguous_pairing'])

    def test_original_image_entry_offsets_every_evidence(self):
        detector = Mock()
        detector.detect.return_value = {'detections':[{'class_id':2,'bbox':[100,200,500,500],'confidence':.9}]}
        image = Image.new('RGB',(800,700),(15,40,160))
        with patch('traffic_agent.direction_reader.read_text', return_value={'lines':[line()]}), \
             patch('traffic_agent.direction_reader.detect_direction_arrows', return_value={'arrows':[arrow()]}):
            report = read_directions(image, detector, panel_recovery=False)
        self.assertEqual(detector.detect.call_args.args[0].size, (800,700))
        sign = report['signs'][0]
        relation = sign['relations'][0]
        self.assertEqual(relation['text_bbox'], [220,300,360,340])
        self.assertEqual(relation['arrow_bbox'], [140,290,190,350])
        self.assertEqual(sign['texts'][0]['quad'][0], [220,300])
        self.assertEqual(sign['arrow_diagnostics']['coordinate_frame'],'crop_local_pixels')
        self.assertEqual(sign['association_diagnostics']['origin_in_original'],[100,200])

    def test_detection_miss_never_calls_ocr(self):
        detector=Mock()
        detector.detect.return_value={'detections':[]}
        with patch('traffic_agent.direction_reader.read_text') as ocr:
            report=read_directions(Image.new('RGB',(800,700)),detector)
        ocr.assert_not_called()
        self.assertEqual(report['status'],'abstain')

    def test_inverse_mapping_uses_all_four_corners(self):
        # x'=x-y: the extrema are the other two corners, not the diagonal.
        inverse=[[1,-1,0],[0,1,0],[0,0,1]]
        mapped=_map_evidence({'bbox':[10,20,40,60]},inverse,100,200)
        self.assertEqual(mapped['bbox'],[50,220,120,260])
        self.assertEqual(mapped['quad'],[[90,220],[120,220],[80,260],[50,260]])

    def test_rectified_observations_return_original_coordinates(self):
        detector=Mock()
        detector.detect.return_value={'detections':[{'class_id':2,'bbox':[100,200,500,500],'confidence':.9}]}
        rectified=Image.new('RGB',(400,300),(15,40,160))
        transform={'image':rectified,'status':'rectified','reason':'supported_panel_quadrilateral',
                   'homography':[[1,0,-10],[0,1,-20],[0,0,1]],
                   'inverse_homography':[[1,0,10],[0,1,20],[0,0,1]],
                   'source_quad':[[10,20],[390,20],[390,290],[10,290]],
                   'rectified_size':[400,300],'direction_frame':'rectified_panel_axes'}
        with patch('traffic_agent.direction_reader.rectify_panel',return_value=transform), \
             patch('traffic_agent.direction_reader.read_text',return_value={'lines':[line()]}), \
             patch('traffic_agent.direction_reader.detect_direction_arrows',return_value={'arrows':[arrow()]}):
            report=read_directions(Image.new('RGB',(800,700)),detector,panel_recovery=False)
        sign=report['signs'][0]
        self.assertEqual(sign['relations'][0]['text_bbox'],[230,320,370,360])
        self.assertEqual(sign['relations'][0]['arrow_bbox'],[150,310,200,370])
        self.assertEqual(sign['texts'][0]['quad'][0],[230,320])
        self.assertEqual(sign['perspective']['source_quad_original'][0],[110,220])
        self.assertEqual(sign['arrow_diagnostics']['coordinate_frame'],'rectified_panel_pixels')
        self.assertEqual(sign['relations'][0]['direction_frame'],'rectified_panel_axes')

    def test_exif_normalization_precedes_detection(self):
        image=Image.new('RGB',(800,600));image.getexif()[274]=6
        detector=Mock();detector.detect.return_value={'detections':[]}
        report=read_directions(image,detector)
        self.assertEqual(report['image_size'],[600,800])
        self.assertEqual(detector.detect.call_args_list[0].args[0].size,(600,800))

    def test_pil_crop_preserves_rgb_contract_for_ocr(self):
        image=Image.new('RGB',(20,20),(230,30,10))
        fake=Mock(return_value=(None, None))
        with patch('traffic_agent.ocr.engine',return_value=fake):
            read_text(image)
        supplied=fake.call_args.args[0]
        self.assertIsInstance(supplied,Image.Image)
        self.assertEqual(supplied.getpixel((0,0)),(230,30,10))


if __name__=='__main__': unittest.main()
