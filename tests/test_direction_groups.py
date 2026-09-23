import unittest
import cv2
import numpy as np
from PIL import Image

from traffic_agent.direction_groups import associate_shared_arrow, associate_banded_rows


def text(text, box, index):
    x,y,r,b=box
    return {'text_id':f't{index}','text':text,'score':.97,'bbox':box,
            'quad':[[x,y],[r,y],[r,b],[x,b]]}


def scene():
    rgb=np.zeros((480,800,3),np.uint8);rgb[:]=[15,40,160]
    rgb[:100]=230
    lines=[text('环城路',[200,20,600,85],0),text('人民路',[40,150,280,210],1),
           text('黄河路',[510,150,750,210],2),text('文化广场',[40,320,280,380],3),
           text('白云',[520,320,730,380],4)]
    points=np.array([[400,300],[445,340],[420,340],[420,420],[380,420],[380,340],[355,340]])
    cv2.fillPoly(rgb,[points],(240,240,240))
    arrow={'arrow_id':'a1','direction':'up','bbox':[355,300,446,421],
           'shape_iou':.94,'direction_margin':.25}
    return rgb,lines,[arrow]


class SharedDirectionTests(unittest.TestCase):
    def test_four_destinations_share_one_arrow_header_is_context(self):
        rgb,lines,arrows=scene()
        relations,reasons,info=associate_shared_arrow(lines,arrows,Image.fromarray(rgb))
        self.assertFalse(reasons)
        self.assertEqual([r['destination'] for r in relations],['人民路','黄河路','文化广场','白云'])
        self.assertEqual({r['arrow_id'] for r in relations},{'a1'})
        self.assertEqual({r['group_id'] for r in relations},{'g1'})
        self.assertEqual(info['excluded_texts'][0]['text'],'环城路')

    def test_extra_road_structure_blocks_all_links(self):
        rgb,lines,arrows=scene()
        cv2.line(rgb,(320,125),(320,285),(240,240,240),14)
        cv2.line(rgb,(290,240),(450,240),(240,240,240),14)
        relations,reasons,_=associate_shared_arrow(lines,arrows,Image.fromarray(rgb))
        self.assertEqual(relations,[])
        self.assertEqual(reasons,['unexplained_road_structure'])

    def test_missing_arrow_and_multiple_arrows_abstain(self):
        rgb,lines,arrows=scene()
        for aa in ([],arrows*2):
            self.assertEqual(associate_shared_arrow(lines,aa,Image.fromarray(rgb))[0],[])

    def test_direction_cannot_be_assigned_to_arbitrary_names(self):
        rgb,lines,arrows=scene()
        lines[2]['bbox']=[510,245,750,290]
        self.assertEqual(associate_shared_arrow(lines,arrows,Image.fromarray(rgb))[1],['shared_group_layout'])

    def test_low_ocr_score_and_crop_edge_fail_closed(self):
        rgb,lines,arrows=scene()
        lines[1]['score']=.6
        self.assertEqual(associate_shared_arrow(lines,arrows,Image.fromarray(rgb))[1],['low_text_confidence'])
        lines[1]['score']=.97;lines[1]['bbox'][0]=1
        self.assertEqual(associate_shared_arrow(lines,arrows,Image.fromarray(rgb))[1],['clipped_text'])

    def test_down_or_lateral_arrow_cannot_explain_two_column_up_layout(self):
        rgb,lines,arrows=scene()
        for direction in ('down','left','right'):
            arrows[0]['direction']=direction
            self.assertEqual(associate_shared_arrow(lines,arrows,Image.fromarray(rgb))[0],[])

    def test_prompt_text_is_not_a_destination(self):
        rgb,lines,arrows=scene();lines[1]['text']='请按导向车道行驶'
        self.assertEqual(associate_shared_arrow(lines,arrows,Image.fromarray(rgb))[1],['prompt_sign'])

    def test_overlapping_ocr_alternatives_are_not_multiple_destinations(self):
        rgb,lines,arrows=scene()
        duplicate=text('民族路',[40,153,280,213],5)
        self.assertEqual(associate_shared_arrow(lines+[duplicate],arrows,Image.fromarray(rgb))[1],['ambiguous_pairing'])

    def test_auxiliary_exit_label_is_not_linked_to_arrow(self):
        rgb,lines,arrows=scene()
        lines.append(text('出口',[300,230,460,280],5))
        relations,reasons,_=associate_shared_arrow(lines,arrows,Image.fromarray(rgb))
        self.assertFalse(reasons)
        self.assertNotIn('出口',[r['destination'] for r in relations])


class BandedDirectionTests(unittest.TestCase):
    def make_scene(self):
        rgb=np.zeros((600,800,3),np.uint8);rgb[:]=[15,40,160]
        cv2.line(rgb,(0,200),(799,200),(240,240,240),8)
        cv2.line(rgb,(0,400),(799,400),(240,240,240),8)
        lines=[text('西',[40,10,80,50],0),text('人民路',[240,80,550,140],1),
               text('和',[250,280,330,340],2),text('田',[510,280,590,340],3),
               text('文化路',[80,480,450,540],4)]
        arrows=[{'arrow_id':f'a{i}','direction':direction,'bbox':box,'shape_iou':.93,'direction_margin':.23}
                for i,(direction,box) in enumerate([('up',[100,70,180,150]),('left',[100,270,180,350]),
                                                     ('right',[610,470,690,550])],1)]
        return rgb,lines,arrows

    def test_visible_bands_support_alternating_arrow_sides_and_split_characters(self):
        rgb,lines,arrows=self.make_scene()
        relations,reasons,_=associate_banded_rows(lines,arrows,Image.fromarray(rgb))
        self.assertFalse(reasons)
        self.assertEqual([(r['destination'],r['direction']) for r in relations],
                         [('人民路','up'),('和田','left'),('文化路','right')])
        self.assertEqual(relations[1]['text_ids'],['t2','t3'])
        self.assertEqual(len(relations[1]['text_segments']),2)

    def test_same_observations_without_physical_dividers_are_not_joined(self):
        rgb,lines,arrows=self.make_scene();rgb[:]=[15,40,160]
        self.assertEqual(associate_banded_rows(lines,arrows,Image.fromarray(rgb))[0],[])

    def test_two_arrows_in_one_band_is_ambiguous(self):
        rgb,lines,arrows=self.make_scene()
        arrows.append(dict(arrows[0],arrow_id='a4'))
        self.assertEqual(associate_banded_rows(lines,arrows,Image.fromarray(rgb))[1],['ambiguous_pairing'])

    def test_low_confidence_character_is_not_completed(self):
        rgb,lines,arrows=self.make_scene();lines[3]['score']=.4
        self.assertEqual(associate_banded_rows(lines,arrows,Image.fromarray(rgb))[1],['low_text_confidence'])

    def test_large_unexplained_junction_rejects_banded_board(self):
        rgb,lines,arrows=self.make_scene()
        cv2.line(rgb,(680,30),(680,175),(240,240,240),15)
        cv2.line(rgb,(600,100),(750,100),(240,240,240),15)
        relations,reasons,_=associate_banded_rows(lines,arrows,Image.fromarray(rgb))
        self.assertEqual(relations,[])
        self.assertEqual(reasons,['unexplained_road_structure'])

    def test_missing_arrow_only_rejects_its_separate_row(self):
        rgb,lines,arrows=self.make_scene()
        relations,reasons,info=associate_banded_rows(lines,arrows[:2],Image.fromarray(rgb))
        self.assertEqual([r['destination'] for r in relations],['人民路','和田'])
        self.assertEqual(reasons,['no_arrow'])
        self.assertEqual(info['unresolved_bands'],[{'band':3,'reason':'no_arrow'}])


if __name__=='__main__':unittest.main()
