import unittest

import cv2
import numpy as np
from PIL import Image,ImageDraw

from traffic_agent.perspective import rectify_panel,transform_points,_geometry_reason


def board(quad=None):
    # Test-only synthetic geometry; the inference API receives pixels only.
    face=Image.new('RGB',(360,400),'#1248b0')
    draw=ImageDraw.Draw(face)
    draw.rectangle((8,8,351,391),outline='white',width=4)
    draw.line((8,180,351,180),fill='white',width=5)
    draw.rectangle((70,65,280,95),fill='white')
    draw.rectangle((70,230,280,260),fill='white')
    quad=np.asarray(quad if quad is not None else [[110,45],[425,90],[490,470],[75,450]],np.float32)
    matrix=cv2.getPerspectiveTransform(np.array([[0,0],[359,0],[359,399],[0,399]],np.float32),quad)
    pixels=cv2.warpPerspective(np.asarray(face),matrix,(560,530),borderValue=(160,160,160))
    return Image.fromarray(pixels)


class PerspectiveTests(unittest.TestCase):
    def test_automatic_rectification_preserves_roundtrip_coordinates(self):
        result=rectify_panel(board())
        self.assertEqual(result['status'],'rectified')
        self.assertEqual(result['direction_frame'],'rectified_panel_axes')
        points=np.array([[150.25,100.75],[280.125,220.5],[420.8,380.25]])
        mapped=transform_points(points,result['homography'])
        recovered=transform_points(mapped,result['inverse_homography'])
        np.testing.assert_allclose(recovered,points,atol=1e-6)
        target=np.asarray(transform_points(result['source_quad'],result['homography']))
        self.assertLess(target[0,0],target[1,0])
        self.assertLess(target[0,1],target[3,1])
        self.assertAlmostEqual(target[0,1],target[1,1],places=3)

    def test_near_frontal_panel_is_not_resampled(self):
        image=board([[90,60],[450,60],[450,460],[90,460]])
        result=rectify_panel(image)
        self.assertEqual(result['status'],'unchanged')
        self.assertEqual(result['reason'],'already_near_frontal')
        np.testing.assert_array_equal(result['image'],image)
        np.testing.assert_array_equal(result['homography'],np.eye(3))

    def test_crop_cutting_through_panel_keeps_identity(self):
        image=board().crop((140,0,560,360))
        result=rectify_panel(image)
        self.assertEqual(result['status'],'unchanged')
        self.assertEqual(result['image'].size,image.size)
        np.testing.assert_array_equal(result['homography'],np.eye(3))
        np.testing.assert_array_equal(result['inverse_homography'],np.eye(3))

    def test_no_color_panel_and_tiny_crop_abstain(self):
        for size in [(500,400),(60,60)]:
            with self.subTest(size=size):
                image=Image.new('RGB',size,'gray')
                result=rectify_panel(image)
                self.assertEqual(result['status'],'unchanged')
                self.assertIsNone(result['source_quad'])
                np.testing.assert_array_equal(result['image'],image)

    def test_inverse_maps_all_four_box_corners(self):
        result=rectify_panel(board())
        box=[[60,60],[160,60],[160,100],[60,100]]
        quad=transform_points(box,result['inverse_homography'])
        self.assertEqual(len(quad),4)
        np.testing.assert_allclose(transform_points(quad,result['homography']),box,atol=1e-6)
        self.assertNotAlmostEqual(quad[0][1],quad[1][1],places=2)

    def test_large_crop_mapping_retains_original_pixel_coordinates(self):
        image=board().resize((1680,1590),Image.Resampling.BILINEAR)
        result=rectify_panel(image)
        self.assertEqual(result['status'],'rectified')
        self.assertLessEqual(max(result['diagnostics']['analysis_size']),1280)
        self.assertGreater(max(point[0] for point in result['source_quad']),1280)
        points=[[450.5,300.25],[1260.75,1140.5]]
        np.testing.assert_allclose(transform_points(transform_points(points,result['homography']),
                                                    result['inverse_homography']),points,atol=1e-6)

    def test_rejects_mirrored_corner_order_and_extreme_shape(self):
        mirrored=np.array([[100,100],[100,400],[400,400],[400,100]],np.float64)
        self.assertEqual(_geometry_reason(mirrored,(500,500)),'mirrored_or_inconsistent_corner_order')
        extreme=np.array([[240,20],[265,20],[475,470],[25,470]],np.float64)
        self.assertIsNotNone(_geometry_reason(extreme,(500,500)))

    def test_transform_rejects_horizon_and_nonfinite_points(self):
        with self.assertRaises(ValueError):
            transform_points([[float('nan'),0]],np.eye(3))
        with self.assertRaises(ValueError):
            transform_points([[0,0]],[[1,0,0],[0,1,0],[1,0,0]])

    def test_supplied_measured_border_has_margin_and_maps_back(self):
        quad=[[110,45],[425,90],[490,470],[75,450]]
        image=board(quad)
        result=rectify_panel(image,source_quad=quad)
        self.assertEqual(result['status'],'rectified')
        self.assertEqual(result['diagnostics']['source'],'yolo_supported_panel_border_proposal')
        points=[[150,130],[450,400]]
        np.testing.assert_allclose(transform_points(transform_points(points,result['homography']),
                                                    result['inverse_homography']),points,atol=1e-6)

    def test_supplied_border_touching_crop_edge_is_rejected(self):
        result=rectify_panel(Image.new('RGB',(560,530),'gray'),
                             source_quad=[[0,45],[425,90],[490,470],[0,450]])
        self.assertEqual(result['status'],'unchanged')
        self.assertIn('clipped',result['reason'])

    def test_slight_vertical_skew_keeps_native_arrow_pixels(self):
        # Slight edge lean is not enough reason to resample otherwise clear text
        # and independent arrow silhouettes.
        image=Image.new('RGB',(600,650),'#1248b0')
        result=rectify_panel(image,source_quad=[[80,40],[550,50],[535,590],[40,585]])
        self.assertEqual(result['status'],'unchanged')
        self.assertEqual(result['reason'],'already_near_frontal')


if __name__=='__main__':
    unittest.main()
