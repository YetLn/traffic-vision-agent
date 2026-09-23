"""Experimental OCR-box refinement supported by intact arrow geometry.

No image pixels are erased, no text is corrected, and no dictionary or expected
place name is used. A revised box is accepted only after local OCR reproduces
the complete original text at high confidence. Coordinates stay crop-local.
This module is opt-in; importing it does not alter the existing OCR pipeline.
"""
from copy import deepcopy
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .direction_geometry import detect_direction_arrows
from .ocr import read_text


MIN_SCORE = .85
MAX_RECHECKS = 4


def _compact(text):
    return ''.join(str(text).split())


def _edge_proposal(line, arrow):
    x, y, r, b = line['bbox']
    ax, ay, ar, ab = arrow['bbox']
    tw, th, aw, ah = r-x, b-y, ar-ax, ab-ay
    if min(tw, th, aw, ah) <= 0:
        return None
    overlap = min(r, ar)-max(x, ax)
    vertical = min(b, ab)-max(y, ay)
    if overlap < min(tw, aw)*.15 or vertical < min(th, ah)*.50:
        return None
    if arrow.get('shape_iou', 0) < .85 or arrow.get('direction_margin', 0) < .16:
        return None
    cx = (ax+ar)/2
    if cx >= r and x < ax < r:
        side, edge = 'right', ax-2
        remaining = edge-x
    elif cx <= x and x < ar < r:
        side, edge = 'left', ar+2
        remaining = r-edge
    else:
        # An arrow-shaped glyph inside the OCR span is not permission to cut
        # characters out of that span.
        return None
    if remaining < tw*.65:
        return None
    return {'side': side, 'edge': edge, 'arrow_id': arrow.get('arrow_id'),
            'arrow_bbox': list(arrow['bbox']), 'shape_iou': arrow['shape_iou'],
            'direction_margin': arrow['direction_margin']}


def _merge_recheck(raw_lines, original, roi):
    left, top, right, bottom = roi
    oy, ob = original['bbox'][1], original['bbox'][3]
    original_height = ob-oy
    relevant = []
    for row in raw_lines:
        x, y, r, b = row['bbox']
        absolute_y, absolute_b = y+top, b+top
        overlap = min(ob, absolute_b)-max(oy, absolute_y)
        if overlap >= .55*min(original_height, b-y):
            relevant.append(row)
    relevant.sort(key=lambda row: (row['bbox'][0], row['bbox'][1]))
    if not relevant:
        return None, 'no_rechecked_text'
    if any(row['score'] < MIN_SCORE for row in relevant):
        return None, 'low_recheck_score'
    text = ''.join(_compact(row['text']) for row in relevant)
    if text != _compact(original['text']):
        return None, 'rechecked_text_changed'
    # OCR duplicates, broken order and two rows must not be silently selected
    # around: every substantial same-row fragment participates in the check.
    centers = [(row['bbox'][1]+row['bbox'][3])/2 for row in relevant]
    heights = [row['bbox'][3]-row['bbox'][1] for row in relevant]
    if max(centers)-min(centers) > min(heights)*.40:
        return None, 'recheck_rows_disagree'
    if any(min(row['bbox'][0], row['bbox'][1], right-left-row['bbox'][2],
               bottom-top-row['bbox'][3]) <= 1 for row in relevant):
        return None, 'rechecked_text_touches_roi_edge'
    points = []
    segments = []
    for row in relevant:
        absolute_quad = [[float(x+left), float(y+top)] for x, y in row['quad']]
        points.extend(absolute_quad)
        segments.append({**deepcopy(row), 'quad': absolute_quad,
                         'bbox': [row['bbox'][0]+left, row['bbox'][1]+top,
                                  row['bbox'][2]+left, row['bbox'][3]+top],
                         'center': [round((row['bbox'][0]+row['bbox'][2])/2+left, 1),
                                    round((row['bbox'][1]+row['bbox'][3])/2+top, 1)]})
    if len(relevant) == 1:
        quad = np.asarray(points, dtype=np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(np.asarray(points, dtype=np.float32)))
        # Preserve measured row tilt; an axis-aligned replacement would hide
        # perspective from the later layout gate.
        quad = np.array([quad[np.argmin(quad.sum(axis=1))],
                         quad[np.argmin(np.diff(quad, axis=1).ravel())],
                         quad[np.argmax(quad.sum(axis=1))],
                         quad[np.argmax(np.diff(quad, axis=1).ravel())]])
    x, y = quad.min(axis=0)
    r, b = quad.max(axis=0)
    return {'text': text, 'score': min(row['score'] for row in relevant),
            'bbox': [round(float(v), 1) for v in (x, y, r, b)],
            'quad': [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
            'segments': segments}, None


def refine_ocr_lines(image, lines):
    """Return copied lines plus transparent local diagnostics; no mutations.

    Original OCR observations are retained under ``ocr_refinement.original``
    for accepted revisions. Unconfirmed proposals leave the line unchanged.
    At most four local OCR calls are made; there is no detector or network call.
    """
    originals = deepcopy(list(lines))
    copied = deepcopy(originals)
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert('RGB')
    else:
        image = image.convert('RGB')
    diagnostics = {'method': 'arrow_edge_exact_text_recheck_v1',
                   'coordinate_frame': 'input_crop_pixels', 'modified_count': 0,
                   'recheck_count': 0, 'max_rechecks': MAX_RECHECKS, 'attempts': []}
    geometry = detect_direction_arrows(image, text_boxes=())
    diagnostics['geometry_method'] = geometry.get('method')
    diagnostics['unmasked_arrow_candidates'] = deepcopy(geometry.get('arrows', []))
    for index, original in enumerate(originals):
        if original.get('ocr_refinement') or original.get('score', 0) < MIN_SCORE:
            continue
        if len(_compact(original.get('text', ''))) < 2:
            continue
        proposals = [proposal for arrow in geometry.get('arrows', [])
                     if (proposal := _edge_proposal(original, arrow)) is not None]
        if not proposals:
            continue
        attempt = {'line_index': index, 'text_id': original.get('text_id'),
                   'original_text': original['text'], 'original_bbox': list(original['bbox']),
                   'proposals': proposals, 'status': 'retained'}
        diagnostics['attempts'].append(attempt)
        if len(proposals) != 1:
            attempt['reason'] = 'multiple_edge_candidates'
            continue
        if diagnostics['recheck_count'] >= MAX_RECHECKS:
            attempt['reason'] = 'recheck_budget_exhausted'
            continue
        x, y, r, b = original['bbox']
        height = b-y
        if min(x, y, image.width-r, image.height-b) <= 3:
            attempt['reason'] = 'original_text_near_image_edge'
            continue
        quad = original.get('quad')
        if quad and abs(math.degrees(math.atan2(quad[1][1]-quad[0][1], quad[1][0]-quad[0][0]))) > 12:
            attempt['reason'] = 'original_row_tilted'
            continue
        proposal = proposals[0]
        left, right = max(0, math.floor(x-height*.15)), min(image.width, math.ceil(r+height*.15))
        if proposal['side'] == 'right':
            right = min(right, math.floor(proposal['edge']))
        else:
            left = max(left, math.ceil(proposal['edge']))
        roi = [left, max(0, math.floor(y-height*.25)), right,
               min(image.height, math.ceil(b+height*.25))]
        attempt['recheck_roi'] = roi
        diagnostics['recheck_count'] += 1
        try:
            raw = read_text(image.crop(roi), min_score=.5)['lines']
        except Exception as exc:
            attempt.update(reason='ocr_recheck_unavailable', error_type=type(exc).__name__)
            continue
        attempt['rechecked_raw_lines'] = deepcopy(raw)
        attempt['rechecked_raw_coordinate_frame'] = 'recheck_roi_pixels'
        revised, reason = _merge_recheck(raw, original, roi)
        if revised is None:
            attempt['reason'] = reason
            continue
        bx, by, br, bb = revised['bbox']
        ax, ay, ar, ab = proposal['arrow_bbox']
        if min(br, ar) > max(bx, ax) and min(bb, ab) > max(by, ay):
            attempt['reason'] = 'rechecked_box_still_overlaps_arrow'
            continue
        if br-bx >= r-x or min(bx, by, image.width-br, image.height-bb) <= 3:
            attempt['reason'] = 'rechecked_box_not_tighter_or_clipped'
            continue
        revised_line = copied[index]
        revised_line.update(bbox=revised['bbox'], quad=revised['quad'],
                            center=[round((bx+br)/2, 1), round((by+bb)/2, 1)],
                            height=round(bb-by, 1), score=min(original['score'], revised['score']))
        revised_line['ocr_refinement'] = {'coordinate_frame': 'input_crop_pixels',
            'original': deepcopy(original),
            'basis': 'intact_arrow_at_span_edge_and_exact_local_ocr_agreement',
            'proposal': proposal, 'recheck_roi': roi, 'rechecked': revised}
        attempt.update(status='refined', reason='exact_text_verified', revised_bbox=revised['bbox'])
        diagnostics['modified_count'] += 1
    return {'lines': copied, 'diagnostics': diagnostics}
