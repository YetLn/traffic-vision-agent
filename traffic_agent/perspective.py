"""Bounded, pixel-only rectification of a complete blue/green sign panel.

No OCR text, annotation corners, detector boxes or expected directions are used.
The estimated four borders must be supported by actual image gradients. A
clipped or uncertain panel remains unchanged. The transform preserves corner
order; it does not infer a mirrored photograph's real-world orientation.
Directions measured after rectification refer to the rectified panel axes.
"""
import math

import cv2
import numpy as np
from PIL import Image


def transform_points(points, homography):
    """Map Nx2 points with a 3x3 homography, retaining subpixel precision."""
    points = np.asarray(points,dtype=np.float64)
    matrix = np.asarray(homography,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or matrix.shape!=(3,3):
        raise ValueError('Expected Nx2 points and a 3x3 homography')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError('Transform inputs must be finite')
    mapped = np.column_stack((points,np.ones(len(points))))@matrix.T
    if np.any(np.abs(mapped[:,2])<1e-9):
        raise ValueError('Point maps to the homography horizon')
    result=mapped[:,:2]/mapped[:,2,None]
    if not np.isfinite(result).all():
        raise ValueError('Transform produced nonfinite coordinates')
    return result.tolist()


def _order_quad(points):
    points=np.asarray(points,dtype=np.float64).reshape(4,2)
    center=points.mean(axis=0)
    order=np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
    points=points[order]
    points=np.roll(points,-int(np.argmin(points.sum(axis=1))),axis=0)
    return points  # TL, TR, BR, BL in the image's downward-positive y axis.


def _geometry_reason(quad,size):
    width,height=size
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
        return 'nonconvex_or_invalid_quad'
    vectors=np.roll(quad,-1,axis=0)-quad
    crosses=np.cross(vectors,np.roll(vectors,-1,axis=0))
    if np.any(crosses<=0):
        return 'mirrored_or_inconsistent_corner_order'
    edge=np.linalg.norm(vectors,axis=1)
    if min(edge)<48:
        return 'panel_too_small'
    area=cv2.contourArea(quad.astype(np.float32))
    if not .15<=area/(width*height)<=.97:
        return 'unsupported_panel_area'
    border=max(1.5,min(width,height)*.002)
    if (quad[:,0].min()<=border or quad[:,1].min()<=border or
            quad[:,0].max()>=width-1-border or quad[:,1].max()>=height-1-border):
        return 'panel_clipped_at_crop_edge'
    if max(edge[0],edge[2])/min(edge[0],edge[2])>2.2 or max(edge[1],edge[3])/min(edge[1],edge[3])>2.2:
        return 'extreme_perspective'
    angles=[]
    for i in range(4):
        a=quad[(i-1)%4]-quad[i];b=quad[(i+1)%4]-quad[i]
        angle=math.degrees(math.acos(float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
        angles.append(angle)
    if min(angles)<45 or max(angles)>135:
        return 'extreme_corner_angle'
    # Reject rotations that could confuse the intended top edge or swap axes.
    horizontal=[abs(math.degrees(math.atan2(vectors[i,1],vectors[i,0]))) for i in (0,2)]
    horizontal=[min(value,abs(180-value)) for value in horizontal]
    if max(horizontal)>35 or vectors[0,0]<=0 or vectors[1,1]<=0:
        return 'unsupported_panel_rotation'
    return None


def _edge_support(quad,distance):
    tolerance=max(2.5,min(distance.shape)*.0055)
    support=[]
    for index in range(4):
        a,b=quad[index],quad[(index+1)%4]
        points=np.rint(a+(b-a)*np.linspace(.05,.95,120)[:,None]).astype(int)
        points[:,0]=np.clip(points[:,0],0,distance.shape[1]-1)
        points[:,1]=np.clip(points[:,1],0,distance.shape[0]-1)
        support.append(float(np.mean(distance[points[:,1],points[:,0]]<=tolerance)))
    return support


def _estimate_quad(rgb, allow_one_weak_edge=False):
    height,width=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),50,120)
    distance=cv2.distanceTransform((edges==0).astype(np.uint8),cv2.DIST_L2,3)
    candidates,rejected=[],[]
    # Independent thresholds avoid joining a desaturated billboard behind a
    # saturated traffic panel. Border support, not text content, selects one.
    for color,low_h,high_h in [('blue',88,140),('green',35,88)]:
        for saturation in (90,120,140,170):
            mask=cv2.inRange(hsv,np.array([low_h,saturation,20],np.uint8),np.array([high_h,255,255],np.uint8))
            kernel=max(3,round(min(width,height)*.005)|1)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            major=[c for c in contours if cv2.contourArea(c)>=width*height*.012]
            if not major:
                continue
            # Include all substantial color components, including separated
            # rows. Rectifying only one convenient row would silently crop data.
            hull=cv2.convexHull(np.vstack(major))
            perimeter=cv2.arcLength(hull,True)
            for epsilon in (.01,.015,.02,.025):
                polygon=cv2.approxPolyDP(hull,perimeter*epsilon,True)
                if len(polygon)!=4:
                    continue
                quad=_order_quad(polygon)
                reason=_geometry_reason(quad,(width,height))
                if reason:
                    rejected.append(reason)
                    continue
                if cv2.contourArea(polygon)<cv2.contourArea(hull)*.90:
                    rejected.append('quadrilateral_drops_visible_boundary')
                    continue
                support=_edge_support(quad,distance)
                supported=(min(support)>=.60 and sum(support)/4>=.75)
                # Only the YOLO-assisted *full-image recovery* path may tolerate
                # one occluded border. Three measured sides still need strong
                # Canny support, and panel_detector then validates seed overlap.
                if allow_one_weak_edge:
                    supported=supported or (sum(value>=.85 for value in support)>=3
                                           and sum(support)/4>=.78)
                if not supported:
                    rejected.append('weak_panel_border_evidence')
                    continue
                candidates.append({'quad':quad,'color':color,'saturation_threshold':saturation,
                                   'edge_support':support,'major_color_components':len(major)})
                break
    diagnostics={'accepted_candidates':len(candidates),'rejected_reasons':sorted(set(rejected))}
    if not candidates:
        return None,diagnostics
    best=max(candidates,key=lambda item:(min(item['edge_support']),sum(item['edge_support'])))
    diagnostics.update({key:value for key,value in best.items() if key!='quad'})
    return best['quad'],diagnostics


def estimate_supported_quad(image, allow_one_weak_edge=False):
    """Estimate a visually backed whole-panel quad in input-image pixels.

    The relaxed option must be combined with actual YOLO support by the caller;
    it is not a stand-alone sign detector or a replacement for rectification.
    """
    image=image.convert('RGB')
    scale=min(1.,1280/max(image.size))
    analysis_size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    quad,diagnostics=_estimate_quad(np.asarray(image.resize(analysis_size,Image.Resampling.BILINEAR)),
                                    allow_one_weak_edge=allow_one_weak_edge)
    diagnostics['analysis_size']=list(analysis_size)
    if quad is None:
        return None,diagnostics
    return (quad/np.array([analysis_size[0]/image.width,analysis_size[1]/image.height])).tolist(),diagnostics


def quad_near_frontal(quad):
    """Keep native pixels when the measured border is almost rectangular."""
    quad=np.asarray(quad,dtype=np.float64)
    vectors=np.roll(quad,-1,axis=0)-quad
    lengths=np.linalg.norm(vectors,axis=1)
    if quad.shape!=(4,2) or min(lengths)<=0:
        return False
    horizontal=max(abs(vectors[0,1])/lengths[0],abs(vectors[2,1])/lengths[2])
    vertical=max(abs(vectors[1,0])/lengths[1],abs(vectors[3,0])/lengths[3])
    perspective=max(max(lengths[0],lengths[2])/min(lengths[0],lengths[2]),
                    max(lengths[1],lengths[3])/min(lengths[1],lengths[3]))
    return (horizontal<math.sin(math.radians(3)) and
            vertical<math.sin(math.radians(5)) and perspective<1.08)


def rectify_panel(crop, source_quad=None):
    """Return image, crop->rectified homography, inverse, source_quad and status.

    ``status`` is ``rectified`` or ``unchanged``. All unchanged results use
    identity matrices and preserve input pixel dimensions. ``source_quad`` is
    TL/TR/BR/BL in input crop pixels, or None when no reliable quad was found.
    Evidence produced on ``image`` must be inverse-mapped before adding the
    original detection crop offset. Preserve quadrilaterals, not just two box
    corners: a perspective transform does not preserve axis-aligned boxes.
    """
    if not isinstance(crop,Image.Image):
        raise ValueError('rectify_panel requires a PIL image crop')
    image=crop.convert('RGB')
    identity=np.eye(3).tolist()
    base={'image':image,'status':'unchanged','reason':'no_reliable_quadrilateral',
          'homography':identity,'inverse_homography':identity,'source_quad':None,
          'rectified_size':list(image.size),'coordinate_frame':'crop_pixels',
          'direction_frame':'crop_axes','diagnostics':{}}
    if min(image.size)<96:
        return {**base,'reason':'crop_too_small'}
    if source_quad is None:
        scale=min(1.,1280/max(image.size))
        analysis_size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
        rgb=np.asarray(image.resize(analysis_size,Image.Resampling.BILINEAR))
        quad,diagnostics=_estimate_quad(rgb)
        diagnostics['analysis_size']=list(analysis_size)
        if quad is not None:
            quad=quad/np.array([analysis_size[0]/image.width,analysis_size[1]/image.height])
    else:
        quad=np.asarray(source_quad,dtype=np.float64)
        if quad.shape!=(4,2):
            raise ValueError('source_quad must have four 2D corners')
        reason=_geometry_reason(quad,image.size)
        if reason:
            return {**base,'reason':'supplied_visual_quad_'+reason,
                    'diagnostics':{'source':'yolo_supported_panel_border_proposal'}}
        diagnostics={'source':'yolo_supported_panel_border_proposal',
                     'analysis_size':list(image.size),'accepted_candidates':1,
                     'rejected_reasons':[]}
    base['diagnostics']=diagnostics
    if quad is None:
        if 'panel_clipped_at_crop_edge' in diagnostics['rejected_reasons']:
            base['reason']='panel_clipped_at_crop_edge'
        return base
    base['source_quad']=quad.tolist()
    vectors=np.roll(quad,-1,axis=0)-quad
    lengths=np.linalg.norm(vectors,axis=1)
    if quad_near_frontal(quad):
        return {**base,'reason':'already_near_frontal'}
    width,height=(lengths[0]+lengths[2])/2,(lengths[1]+lengths[3])/2
    if not .20<=width/height<=5 or width*height>image.width*image.height*1.5:
        return {**base,'reason':'unsupported_rectified_shape'}
    output_scale=min(1.,1600/max(width,height))
    pad=4
    output_size=(max(2,round(width*output_scale))+2*pad,max(2,round(height*output_scale))+2*pad)
    destination=np.array([[pad,pad],[output_size[0]-1-pad,pad],
                          [output_size[0]-1-pad,output_size[1]-1-pad],[pad,output_size[1]-1-pad]],np.float32)
    matrix=cv2.getPerspectiveTransform(quad.astype(np.float32),destination)
    try:
        inverse=np.linalg.inv(matrix)
        center=quad.mean(axis=0)
        probes=np.asarray(transform_points([center,center+[1,0],center+[0,1]],matrix))
        orientation=np.cross(probes[1]-probes[0],probes[2]-probes[0])
        if not np.isfinite(matrix).all() or not np.isfinite(inverse).all() or orientation<=0:
            return {**base,'reason':'unstable_or_mirrored_transform'}
    except (np.linalg.LinAlgError,ValueError):
        return {**base,'reason':'unstable_or_mirrored_transform'}
    rectified=cv2.warpPerspective(np.asarray(image),matrix,output_size,
                                 flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))
    diagnostics['output_scale']=output_scale
    return {**base,'image':Image.fromarray(rectified),'status':'rectified',
            'reason':'supported_panel_quadrilateral','homography':matrix.tolist(),
            'inverse_homography':inverse.tolist(),'rectified_size':list(output_size),
            'coordinate_frame':'rectified_panel_pixels','direction_frame':'rectified_panel_axes'}
