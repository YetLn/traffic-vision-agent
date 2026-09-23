"""Original-image panel proposals with explicit provenance.

The checkpoint was trained largely on smaller signs. A second, genuinely YOLO
pass places the *whole* image in a larger gray canvas to expose large close-up
signs at a familiar scale. Coordinates are mapped back to the original pixels.
Blue/green rectangular regions optionally complete panel borders or provide
UNCONFIRMED geometry proposals. Geometry confidence is never invented as YOLO
confidence. No labels, gold boxes, filenames or expected destinations are inputs.
"""

import copy
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from .perspective import estimate_supported_quad, quad_near_frontal


def _area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def _intersection(a, b):
    return max(0, min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def _iou(a, b):
    inter = _intersection(a,b)
    return inter/max(1, _area(a)+_area(b)-inter)


def _clip(box, size):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return None
    width, height = size
    value = [max(0,min(width,float(box[0]))), max(0,min(height,float(box[1]))),
             max(0,min(width,float(box[2]))), max(0,min(height,float(box[3])))]
    return [round(v,2) for v in value] if _area(value) > 0 else None


def color_panel_proposals(image):
    """Return rectangular blue/green color regions; these are NOT classified signs."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    scale = min(1.0, 1200/max(image.size))
    width, height = max(1,round(image.width*scale)), max(1,round(image.height*scale))
    rgb = np.asarray(image.resize((width,height), Image.Resampling.BILINEAR))
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # Highly saturated backgrounds suppress sky; white headers remain inside an
    # external blue border. Closing joins small border gaps, not distant panels.
    colors = [('blue',(88,95,25),(140,255,255)), ('green',(35,85,25),(88,255,255))]
    proposals = []
    for name, low, high in colors:
        raw = cv2.inRange(hsv,np.array(low,np.uint8),np.array(high,np.uint8))
        kernel = max(3, int(round(min(width,height)*.008))|1)
        mask = cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if min(w/scale,h/scale)<100 or w*h/(scale*scale)<24000 or not .30 <= w/h <= 7:
                continue
            if w*h < width*height*.006 or w*h > width*height*.96:
                continue
            rectangularity = area/(w*h)
            if rectangularity < .64:
                continue
            polygon = cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
            if not 4 <= len(polygon) <= 8:
                continue
            color_fraction = float(np.count_nonzero(raw[y:y+h,x:x+w]))/(w*h)
            if color_fraction < .22:
                continue
            # Ignore almost solid fields (sky/building walls): a road panel must
            # contain some light foreground. This is only a proposal filter.
            region = hsv[y:y+h,x:x+w]
            light_fraction = float(np.count_nonzero((region[:,:,1]<90)&(region[:,:,2]>120)))/(w*h)
            if not .015 <= light_fraction <= .70:
                continue
            pad = max(2,round(min(w,h)*.008))
            bbox = _clip([(x-pad)/scale,(y-pad)/scale,(x+w+pad)/scale,(y+h+pad)/scale],image.size)
            proposals.append({'bbox':bbox,'geometry':{'color':name,
                'rectangularity':round(rectangularity,4),'color_fraction':round(color_fraction,4),
                'light_fraction':round(light_fraction,4),'analysis_scale':scale}})
    kept = []
    for candidate in sorted(proposals,key=lambda p:_area(p['bbox']),reverse=True):
        if any(_intersection(candidate['bbox'],p['bbox'])/max(1,_area(candidate['bbox']))>.88 for p in kept):
            continue
        kept.append(candidate)
    return kept


def _yolo_rows(result, size, source, transform=None, rejected=None):
    rows = []
    for detection in result['detections']:
        if detection['class_id'] != 2:
            continue
        box = detection['bbox']
        if transform:
            offset_x,offset_y,scale = transform
            box = [(box[0]-offset_x)/scale,(box[1]-offset_y)/scale,
                   (box[2]-offset_x)/scale,(box[3]-offset_y)/scale]
            # A canvas-border detection must not become a plausible sign after
            # silently clipping away its unsupported area in the gray padding.
            tolerance = max(2,.02*min(box[2]-box[0],box[3]-box[1]))
            if box[0] < -tolerance or box[1] < -tolerance or box[2] > size[0]+tolerance or box[3] > size[1]+tolerance:
                if rejected is not None:
                    rejected.append({'source':source,'bbox':box,'reason':'extends_into_padding'})
                continue
        box = _clip(box,size)
        if box is None:
            continue
        support = {'source':source,'bbox':box,'confidence':detection['confidence']}
        rows.append({'class_id':2,'class_name':'point-l','bbox':box,
                     'confidence':detection['confidence'],'confidence_kind':'yolo_model_confidence',
                     'proposal_source':source,'seed_yolo_bbox':box,'yolo_supported':True,'yolo_support':[support]})
    return rows


def detect_panels(image, detector, conf=.25, allow_geometry_only=False):
    """Return panel hypotheses in EXIF-normalized original-image pixels.

    Always runs real YOLO on the original image and a context-padded image.
    Geometry-only proposals are DISABLED by default. If explicitly enabled for
    diagnosis, they have confidence=None and yolo_supported=False.
    The optional geometry does not turn an unconfirmed rectangle into a verified
    traffic sign; callers must preserve proposal_source in UI/results.
    """
    if image is None:
        raise ValueError('An original road image is required')
    if not 0 < conf <= 1:
        raise ValueError('Confidence threshold must be in (0,1]')
    start = time.perf_counter()
    image = ImageOps.exif_transpose(image).convert('RGB')
    original = detector.detect(image,conf)
    seeds = _yolo_rows(original,image.size,'yolo_original')
    # Keep the auxiliary canvas within 16.8 MP even for large input photographs.
    scale = min(1.0,2048/max(image.size))
    small = image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.BILINEAR)
    side = 2*max(small.size)
    offset_x,offset_y = (side-small.width)//2,(side-small.height)//2
    canvas = Image.new('RGB',(side,side),(114,114,114))
    canvas.paste(small,(offset_x,offset_y))
    padded = detector.detect(canvas,conf)
    rejected = []
    padded_rows = _yolo_rows(padded,image.size,'yolo_padded',(offset_x,offset_y,scale),rejected)
    seeds.extend(copy.deepcopy(padded_rows))
    geometry = color_panel_proposals(image)
    # A close oblique sign can be split into YOLO row boxes while blue contour
    # rectangularity drops below the box-proposal threshold. Recover the board
    # only when three actual border edges and a large YOLO seed agree. The
    # fourth border may be occluded; the polygon remains a measured proposal.
    if seeds:
        quad,quad_info=estimate_supported_quad(image,allow_one_weak_edge=True)
        if quad is not None and not quad_near_frontal(quad):
            q=np.asarray(quad,dtype=np.float32)
            x1,y1=q.min(axis=0);x2,y2=q.max(axis=0)
            candidate_box=_clip([x1-2,y1-2,x2+2,y2+2],image.size)
            support_mask=np.zeros((image.height,image.width),np.uint8)
            cv2.fillConvexPoly(support_mask,np.round(q).astype(np.int32),1)
            seed_support=[]
            for seed in seeds:
                sx,sy,sr,sb=[int(round(v)) for v in seed['bbox']]
                coverage=float(support_mask[sy:sb,sx:sr].mean()) if sr>sx and sb>sy else 0.
                ratio=_area(candidate_box)/max(1,_area(seed['bbox']))
                if (.90<=ratio<=3.3 and coverage>=.55 and
                    _intersection(candidate_box,seed['bbox'])/max(1,_area(seed['bbox']))>=.80):
                    seed_support.append({'bbox':seed['bbox'],'coverage':round(coverage,4),
                                         'candidate_to_seed_area_ratio':round(ratio,4)})
            if seed_support and not any(_iou(candidate_box,p['bbox'])>.80 for p in geometry):
                geometry.append({'bbox':candidate_box,'geometry':{
                    'source':'three_or_four_supported_panel_borders',
                    'source_quad':[[round(float(x),1),round(float(y),1)] for x,y in quad],
                    'edge_support':quad_info['edge_support'],
                    'seed_support':seed_support}})
    panels, assigned, deduplicated = [], set(), []
    for candidate in geometry:
        box = candidate['bbox']
        if box[0]<=1 or box[1]<=1 or box[2]>=image.width-1 or box[3]>=image.height-1:
            rejected.append({'source':'color_geometry','bbox':box,'reason':'touches_image_edge'})
            continue
        supports = []
        for index, seed in enumerate(seeds):
            if index in assigned:
                continue
            overlap = _intersection(box,seed['bbox'])
            # A geometry box may complete a truncated YOLO seed, but must not
            # shrink a larger existing YOLO box or use an unrelated tiny sign.
            is_quad=candidate['geometry'].get('source')=='three_or_four_supported_panel_borders'
            max_ratio=3.3 if is_quad else 2.5
            min_candidate_cover=.28 if is_quad else .40
            if .90<=_area(box)/max(1,_area(seed['bbox']))<=max_ratio and overlap/max(1,_area(seed['bbox']))>=.80 and overlap/max(1,_area(box))>=min_candidate_cover:
                supports.extend(seed['yolo_support'])
                assigned.add(index)
        if supports:
            best = max(supports,key=lambda row:row['confidence'])
            panels.append({'class_id':2,'class_name':'point-l','bbox':box,
                'confidence':best['confidence'],'confidence_kind':'supporting_yolo_confidence_not_geometry_probability',
                'proposal_source':'geometry_with_yolo_support','yolo_supported':True,
                'seed_yolo_bbox':best['bbox'],'yolo_support':supports,'geometry':candidate['geometry']})
        elif allow_geometry_only:
            panels.append({'class_id':2,'class_name':'point-l','bbox':box,
                'confidence':None,'confidence_kind':'not_applicable_geometry_only',
                'proposal_source':'color_geometry','yolo_supported':False,'seed_yolo_bbox':None,
                'yolo_support':[],'geometry':candidate['geometry']})
        else:
            rejected.append({'source':'color_geometry','bbox':box,'reason':'no_yolo_support'})
    # Remaining YOLO candidates are merged by containment so a full panel takes
    # precedence over a row-sized subbox, while preserving both observations.
    for index,seed in sorted(enumerate(seeds),key=lambda row:_area(row[1]['bbox']),reverse=True):
        if index in assigned:
            continue
        containing = next((p for p in panels if _intersection(seed['bbox'],p['bbox'])/max(1,_area(seed['bbox']))>=.85),None)
        if containing:
            containing['yolo_support'].extend(seed['yolo_support'])
            deduplicated.append({'source':seed['proposal_source'],'bbox':seed['bbox'],
                                 'kept_bbox':containing['bbox'],'reason':'contained_duplicate'})
            if not containing['yolo_supported']:
                containing.update(yolo_supported=True,seed_yolo_bbox=seed['bbox'],
                    confidence=seed['confidence'],proposal_source='geometry_with_yolo_support',
                    confidence_kind='supporting_yolo_confidence_not_geometry_probability')
            continue
        panels.append(seed)
    panels.sort(key=lambda p:(p['bbox'][1],p['bbox'][0]))
    return {'detections':panels,'total':len(panels),'counts':{'point-l':len(panels)} if panels else {},
        'image_size':list(image.size),'threshold':conf,'elapsed_ms':round((time.perf_counter()-start)*1000,1),
        'diagnostics':{'method':'yolo_original_plus_context_padding_and_optional_color_geometry_v1',
            'coordinate_frame':'exif_normalized_original_pixels','yolo_passes':2,
            'original_yolo_detections':original['detections'],
            'padded_yolo_detections':padded_rows,
            'padded_transform':{'scale':scale,'offset':[offset_x,offset_y],'canvas_size':[side,side]},
            'geometry_candidates':geometry,
            'allow_geometry_only':allow_geometry_only,'rejected_proposals':rejected,
            'deduplicated_proposals':deduplicated,
            'geometry_only_count':sum(not p['yolo_supported'] for p in panels),
            'limitations':'Geometry is an unconfirmed colored panel proposal. Scores from supporting YOLO do not calibrate the expanded geometry box.'}}
