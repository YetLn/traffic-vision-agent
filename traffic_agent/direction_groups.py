"""Conservative one-arrow/multiple-destination groups on one colored panel.

No OCR correction, place dictionary, expected labels or gold boxes are used.
Shared direction is only allowed for a single standalone arrow, a coherent text
layout and no unexplained large bright structure in the colored content region.
"""
import math
import re

import cv2
import numpy as np

from .sign_text import is_prompt_text


def horizontal_separators(image):
    rgb=np.asarray(image.convert('RGB'))
    height,width=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,40,100)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    pale=(hsv[:,:,1]<100)&(hsv[:,:,2]>110)
    segments=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(35,width//16),
        minLineLength=int(width*.35),maxLineGap=int(width*.08))
    values=[]
    for segment in (() if segments is None else segments[:,0]):
        x,y,r,b=[int(v) for v in segment]
        angle=abs(math.degrees(math.atan2(b-y,r-x)))
        angle=min(angle,abs(180-angle))
        center=(y+b)/2
        if angle <= 8 and height*.10 < center < height*.90 and r!=x:
            # A divider must span most of the panel in pale pixels. Long text
            # underlines, seams and folds alone cannot create a semantic row.
            xs=np.arange(int(width*.07),int(width*.96))
            ys=np.rint(y+(b-y)*(xs-x)/(r-x)).astype(int)
            support=np.zeros(len(xs),bool)
            for delta in range(-5,6):
                valid=(ys+delta>=0)&(ys+delta<height)
                support[valid] |= pale[(ys+delta)[valid],xs[valid]]
            if support.mean()>=.80:
                values.append(center)
    groups=[]
    for value in sorted(values):
        if groups and value-np.mean(groups[-1]) < height*.035:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(float(np.mean(group)),1) for group in groups]


def associate_banded_rows(lines, arrows, image):
    """Use visible horizontal dividers to separate independently labeled rows.

    The arrow can be at either end of each row. Only single-character OCR
    fragments on the same baseline may be joined; original segment boxes remain
    explicit evidence. No dictionary-based completion or character correction.
    """
    separators=horizontal_separators(image)
    info={'method':'separated_arrow_rows_v1','separator_y':separators}
    if not 1<=len(separators)<=4 or len(arrows)<2:
        return [],['no_verified_row_bands'],info
    if any(is_prompt_text(t['text']) for t in lines):
        return [],['prompt_sign'],info
    height=image.height;width=image.width
    boundaries=[0]+separators+[height]
    relations=[]; failures=[]
    info['unresolved_bands']=[]
    def reject(band, reason):
        failures.append(reason)
        info['unresolved_bands'].append({'band':band,'reason':reason})
    all_texts=[t for t in lines if is_destination_candidate(t['text'],allow_single=True)]
    for band,(top,bottom) in enumerate(zip(boundaries,boundaries[1:]),1):
        aa=[a for a in arrows if top<_box_center(a['bbox'])[1]<bottom]
        if len(aa)!=1:
            reject(band,'no_arrow' if not aa else 'ambiguous_pairing');continue
        arrow=aa[0];ay=_box_center(arrow['bbox'])[1]
        candidates=[t for t in all_texts if top<_box_center(t['bbox'])[1]<bottom and
                    abs(_box_center(t['bbox'])[1]-ay)<=max(t['bbox'][3]-t['bbox'][1],arrow['bbox'][3]-arrow['bbox'][1])*.55]
        if not candidates or (len(candidates)>1 and not all(len(t['text'])==1 for t in candidates)):
            reject(band,'ambiguous_pairing');continue
        if any(t['score']<.85 or t['bbox'][3]-t['bbox'][1]<16 for t in candidates):
            reject(band,'low_text_confidence');continue
        candidates.sort(key=lambda t:t['bbox'][0])
        if len(candidates)>1:
            typical=float(np.median([t['bbox'][3]-t['bbox'][1] for t in candidates]))
            if any(b['bbox'][0]-a['bbox'][2]>typical*4 for a,b in zip(candidates,candidates[1:])):
                reject(band,'ambiguous_pairing');continue
        destination=''.join(t['text'] for t in candidates)
        if len(destination)<2:
            reject(band,'no_text');continue
        tb=[min(t['bbox'][0] for t in candidates),min(t['bbox'][1] for t in candidates),
            max(t['bbox'][2] for t in candidates),max(t['bbox'][3] for t in candidates)]
        if min(tb[0],tb[1],width-tb[2],height-tb[3])<=3:
            reject(band,'clipped_text');continue
        if any(t.get('quad') and abs(math.degrees(math.atan2(t['quad'][1][1]-t['quad'][0][1],t['quad'][1][0]-t['quad'][0][0])))>12
               for t in candidates):
            reject(band,'tilted_text');continue
        if arrow['direction'] not in ('up','left','right'):
            reject(band,'unsupported_direction');continue
        row_top=max(0,int(top)+3); row_bottom=min(height,int(bottom)-3)
        def local(item):
            x,y,r,b=item['bbox'];return {**item,'bbox':[x,y-row_top,r,b-row_top]}
        row_lines=[local(t) for t in lines if top<_box_center(t['bbox'])[1]<bottom]
        structure=has_unexplained_structure(image.crop((0,row_top,width,row_bottom)),row_lines,[local(arrow)])
        if structure:
            info.setdefault('unexplained_by_band',[]).append({'band':band,'coordinate_frame':'band_local_pixels',
                'origin_in_crop':[0,row_top],'structures':structure})
            return [],['unexplained_road_structure'],info
        relations.append({'destination':destination,'direction':arrow['direction'],
            'direction_label':{'up':'直行','left':'左转','right':'右转'}[arrow['direction']],
            'text_id':'+'.join(t['text_id'] for t in candidates),'text_ids':[t['text_id'] for t in candidates],
            'text_segments':[{'text_id':t['text_id'],'text':t['text'],'bbox':t['bbox']} for t in candidates],
            'arrow_id':arrow['arrow_id'],'text_bbox':tb,'arrow_bbox':arrow['bbox'],
            'ocr_score':min(t['score'] for t in candidates),'shape_iou':arrow['shape_iou'],
            'direction_margin':arrow['direction_margin'],'group_id':f'row{band}',
            'association_rule':'visible_divider_unique_arrow_row'})
    return relations,sorted(set(failures)),info


def _box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def _chinese(text):
    return bool(re.fullmatch(r'[\u4e00-\u9fff·]{2,12}', re.sub(r'\s+', '', text)))


def is_destination_candidate(text, allow_single=False):
    value=re.sub(r'\s+','',text)
    minimum=1 if allow_single else 2
    return bool(re.fullmatch(r'[\u4e00-\u9fff·]{'+str(minimum)+r',12}',value)) and not any(
        word in value for word in ('方向','出口','入口','前方','距离','海拔','美篇','微信','公众号'))


def colored_text_region(image, lines):
    """Keep names on the colored field; preserve white-header text as context."""
    rgb = np.asarray(image.convert('RGB'))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = ((hsv[:,:,0] >= 40) & (hsv[:,:,0] <= 140) &
               (hsv[:,:,1] >= 65) & (hsv[:,:,2] >= 35))
    height, width = colored.shape
    eligible, excluded = [], []
    for line in lines:
        if not _chinese(line['text']):
            continue
        if not is_destination_candidate(line['text']):
            excluded.append({'text_id':line['text_id'],'text':line['text'],'reason':'auxiliary_text'})
            continue
        x,y,r,b = line['bbox']
        crop = colored[max(0,int(y)):min(height,int(math.ceil(b))),
                       max(0,int(x)):min(width,int(math.ceil(r)))]
        fraction = float(crop.mean()) if crop.size else 0.0
        if fraction >= .20:
            eligible.append(line)
        else:
            excluded.append({'text_id':line['text_id'], 'text':line['text'],
                             'reason':'outside_colored_destination_field',
                             'colored_fraction':round(fraction,3)})
    return eligible, excluded, colored


def _unexplained_symbols(image, colored, lines, arrows):
    """Return unexplained substantial bright structures, never arrow candidates.

    OCR/arrow pixels, verified divider strokes and the perimeter of an actual
    enclosing frame are accounted for before checking residual structures. This
    residual mask is used ONLY to reject a layout, never to detect arrows. Thin
    small supplementary script adjacent to a recognized text row is allowed;
    a large junction/cross remains an unexplained structure even if it touches
    a known frame or OCR box.
    """
    rgb = np.asarray(image.convert('RGB'))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = colored.shape
    rows = np.flatnonzero(colored.mean(axis=1) > .25)
    cols = np.flatnonzero(colored.mean(axis=0) > .25)
    if not len(rows) or not len(cols):
        return [{'reason': 'no_coherent_colored_field'}]
    runs = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    rows = max(runs, key=len)
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]+1), int(rows[0]), int(rows[-1]+1)
    field_width, field_height = x1-x0, y1-y0
    field_edge = max(8, min(field_width, field_height)*.065)
    bright = ((hsv[:, :, 1] < 145) & (hsv[:, :, 2] > 120)).astype(np.uint8)
    bright[:y0] = 0
    bright[y1:] = 0
    bright[:, :x0] = 0
    bright[:, x1:] = 0
    # Convex interiors of substantial colored background components locate
    # the actual board surface, excluding pale outer frames and background.
    # Fill the hull (including holes) so white road diagrams remain inside;
    # do not use the colored mask itself to erase their white pixels.
    field = colored.astype(np.uint8)
    field[:y0] = 0
    field[y1:] = 0
    field[:, :x0] = 0
    field[:, x1:] = 0
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max((cv2.contourArea(contour) for contour in contours), default=0)
    interior = np.zeros_like(bright)
    for contour in contours:
        if cv2.contourArea(contour) >= max(100, field_width*field_height*.03, largest*.15):
            cv2.fillConvexPoly(interior, cv2.convexHull(contour), 1)
    interior = cv2.erode(interior, np.ones((5, 5), np.uint8))
    bright &= interior
    known = np.zeros((height, width), np.uint8)
    for item in list(lines)+list(arrows):
        x, y, r, b = item['bbox']
        pad = max(3, min(16, int(math.ceil((b-y)*.12)))) if 'text' in item else 3
        field_patch = colored[max(0, int(y)):min(height, int(math.ceil(b))),
                              max(0, int(x)):min(width, int(math.ceil(r)))]
        if 'text' in item and field_patch.size and field_patch.mean() < .20:
            # A white-backed heading/compass cell is context, including a
            # modest margin around its OCR glyphs, not a white route map.
            pad = max(3, int((b-y)*.25))
        cv2.rectangle(known, (max(0, int(x)-pad), max(0, int(y)-pad)),
                      (min(width-1, int(math.ceil(r))+pad), min(height-1, int(math.ceil(b))+pad)), 1, -1)
    # Account for only the perimeter pixels of enclosing frames. Never discard
    # the whole enclosing component: an attached road diagram must survive.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if w < field_width*.85 or h < field_height*.85:
            continue
        yy, xx = np.nonzero(labels[y:y+h, x:x+w] == i)
        at_perimeter = ((xx+x < x0+field_edge) | (xx+x > x1-field_edge) |
                        (yy+y < y0+field_edge) | (yy+y > y1-field_edge))
        if at_perimeter.mean() > .55:
            known[y+yy[at_perimeter], x+xx[at_perimeter]] = 1
    # A cropped row does not contain a complete enclosing frame. Account for
    # long, almost straight pale strokes along its outer edges individually.
    # Only the stroke itself is covered; adjacent junction arms remain visible.
    border_edges = cv2.Canny(bright*255, 40, 100)
    border_segments = cv2.HoughLinesP(border_edges, 1, np.pi/180,
        threshold=max(25, int(min(field_width, field_height)*.3)),
        minLineLength=max(30, int(min(field_width, field_height)*.55)), maxLineGap=12)
    border_width = max(8, min(36, int(min(field_width, field_height)*.10)))
    for segment in (() if border_segments is None else border_segments[:, 0]):
        sx, sy, ex, ey = [int(v) for v in segment]
        dx, dy = abs(ex-sx), abs(ey-sy)
        mx, my = (sx+ex)/2, (sy+ey)/2
        vertical = (dy >= field_height*.60 and dx <= dy*.18 and
                    (mx < x0+field_width*.10 or mx > x1-field_width*.10))
        horizontal = (dx >= field_width*.60 and dy <= dx*.18 and
                      (my < y0+field_height*.10 or my > y1-field_height*.10))
        if vertical or horizontal:
            cv2.line(known, (sx, sy), (ex, ey), 1, border_width)
    # Separators have independently verified pale support across >=80% of the
    # width. These are panel boundaries, not independent destination arrows.
    divider_pad = max(6, int(height*.025))
    for center in horizontal_separators(image):
        top, bottom = max(0, int(center)-divider_pad), min(height, int(center)+divider_pad+1)
        known[top:bottom, :] = 1
    residual = (bright & (known == 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    references = [line for line in lines if is_destination_candidate(line['text'], allow_single=True)
                  and line.get('score', 0) >= .85]
    typical_height = float(np.median([line['bbox'][3]-line['bbox'][1] for line in references])) if references else 0
    area_limit = max(100, field_width*field_height*.003)
    unknown = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < area_limit:
            continue
        component = labels[y:y+h, x:x+w] == i
        at_edge = x <= x0+6 or y <= y0+6 or x+w >= x1-6 or y+h >= y1-6
        if at_edge and min(w, h) <= 8:
            continue
        if min(w, h) < 10 and not (w > field_width*.65 or h > field_height*.65):
            continue
        # A long, thin supplementary script row can lack OCR (e.g. Uyghur),
        # while the main Chinese names are recognized. Require small strokes,
        # a small height relative to those names, and row-adjacent placement.
        auxiliary = False
        if typical_height and h <= typical_height*.80 and area/(w*h) <= .55:
            thickness = cv2.distanceTransform(np.pad(component.astype(np.uint8),1), cv2.DIST_L2, 3)[1:-1,1:-1]
            stroke90 = float(np.percentile(thickness[component], 90))*2
            modest_ink = area <= typical_height*typical_height*.18 or w >= h*3
            if modest_ink and stroke90 <= typical_height*.18:
                for line in references:
                    tx, ty, tr, tb = line['bbox']
                    th = tb-ty
                    overlap = min(x+w, tr)-max(x, tx)
                    close_above = ty-th*.8 <= y+h <= ty+th*.25
                    close_below = tb-th*.25 <= y <= tb+th*.8
                    if overlap >= min(w, tr-tx)*.25 and (close_above or close_below):
                        auxiliary = True
                        break
        if auxiliary:
            continue
        unknown.append({'bbox': [x, y, x+w, y+h], 'area': area,
                        'reason': 'unaccounted_bright_structure'})
    return unknown


def has_unexplained_structure(image, lines, arrows):
    """Return a list of unexplained structures in crop-local pixels (empty=none).

    This gate only rejects unsupported visible structure; an empty result is
    not proof of a destination/arrow relationship. Callers must still check
    grouping, OCR quality, clipping and independently detected arrow geometry.
    """
    _, _, colored = colored_text_region(image, lines)
    return _unexplained_symbols(image, colored, lines, arrows)


def associate_shared_arrow(lines, arrows, image):
    """Return relations/reasons/diagnostics for a single arrow's text group.

    Scope: 2-6 names in one column beside an arrow, or 2 aligned columns with
    an UP arrow between them at the bottom. A single arrow in an arbitrary
    layout does not authorize assigning its direction to every visible name.
    """
    diagnostics={'method':'shared_arrow_colored_group_v1', 'excluded_texts':[]}
    if len(arrows) != 1:
        return [], ['shared_arrow_count'], diagnostics
    if any(is_prompt_text(line['text']) for line in lines):
        return [], ['prompt_sign'], diagnostics
    candidates, excluded, colored = colored_text_region(image,lines)
    diagnostics['excluded_texts']=excluded
    if not 2 <= len(candidates) <= 6:
        return [], ['shared_group_layout'], diagnostics
    for index,a in enumerate(candidates):
        for b in candidates[:index]:
            ab,bb=a['bbox'],b['bbox']
            overlap=max(0,min(ab[2],bb[2])-max(ab[0],bb[0]))*max(0,min(ab[3],bb[3])-max(ab[1],bb[1]))
            area=min((ab[2]-ab[0])*(ab[3]-ab[1]),(bb[2]-bb[0])*(bb[3]-bb[1]))
            if overlap/max(1,area)>.20:
                return [],['ambiguous_pairing'],diagnostics
    if any(line['score']<.85 or line['bbox'][3]-line['bbox'][1]<16 for line in candidates):
        return [], ['low_text_confidence'], diagnostics
    width,height=image.size
    for line in candidates:
        x,y,r,b=line['bbox']
        if min(x,y,width-r,height-b)<=3:
            return [], ['clipped_text'], diagnostics
        quad=line.get('quad')
        if quad and abs(math.degrees(math.atan2(quad[1][1]-quad[0][1],quad[1][0]-quad[0][0])))>12:
            return [], ['tilted_text'], diagnostics
    arrow=arrows[0]
    if arrow['direction'] not in ('up','left','right'):
        return [], ['unsupported_direction'], diagnostics
    ab=arrow['bbox']; ax,ay=_box_center(ab)
    median_height=float(np.median([line['bbox'][3]-line['bbox'][1] for line in candidates]))
    if not .5 <= (ab[3]-ab[1])/median_height <= 4:
        return [], ['shared_group_layout'], diagnostics
    starts=[line['bbox'][0] for line in candidates]
    texts_y=[_box_center(line['bbox'])[1] for line in candidates]
    # All rows in a single, non-overlapping label column beside the arrow.
    one_column=(max(starts)-min(starts) <= width*.12 and
                (ab[2] < min(line['bbox'][0] for line in candidates) or
                 ab[0] > max(line['bbox'][2] for line in candidates)) and
                min(texts_y)-median_height <= ay <= max(texts_y)+median_height)
    # Alternatively a symmetric destination grid around an up-arrow.
    left=[line for line in candidates if line['bbox'][2] < ax]
    right=[line for line in candidates if line['bbox'][0] > ax]
    two_columns=False
    if arrow['direction']=='up' and len(left)==len(right) and len(left)>=1 and len(left)+len(right)==len(candidates):
        left.sort(key=lambda line:_box_center(line['bbox'])[1])
        right.sort(key=lambda line:_box_center(line['bbox'])[1])
        same_rows=all(abs(_box_center(a['bbox'])[1]-_box_center(b['bbox'])[1])<=median_height*.55 for a,b in zip(left,right))
        aligned=all(max(t['bbox'][0] for t in col)-min(t['bbox'][0] for t in col)<=width*.12 for col in (left,right))
        bottom_arrow=abs(ay-max(texts_y)) <= max(median_height,ab[3]-ab[1])*.75
        two_columns=same_rows and aligned and bottom_arrow
    ordered_y=sorted(set(round(y/median_height) for y in texts_y))
    if not (one_column or two_columns) or len(ordered_y)>4:
        return [], ['shared_group_layout'], diagnostics
    unknown=_unexplained_symbols(image,colored,lines,arrows)
    diagnostics['unexplained_structures']=unknown
    if unknown:
        return [], ['unexplained_road_structure'], diagnostics
    diagnostics.update(layout='one_column' if one_column else 'two_columns_bottom_up',
                       group_id='g1',text_ids=[line['text_id'] for line in candidates],arrow_id=arrow['arrow_id'])
    names={'up':'直行','left':'左转','right':'右转'}
    relations=[{'destination':line['text'],'direction':arrow['direction'],
                'direction_label':names[arrow['direction']], 'text_id':line['text_id'],
                'arrow_id':arrow['arrow_id'],'group_id':'g1', 'text_bbox':line['bbox'],
                'arrow_bbox':arrow['bbox'], 'ocr_score':line['score'],
                'shape_iou':arrow['shape_iou'],'direction_margin':arrow['direction_margin'],
                'association_rule':'single_arrow_shared_destination_group'} for line in candidates]
    return relations, [], diagnostics
