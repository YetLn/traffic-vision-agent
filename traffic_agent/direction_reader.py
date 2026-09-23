"""Experimental evidence-first reading of simple horizontal direction rows.

YOLO always runs on the original road image. Label boxes are never accepted by
this public entry point. Directions are sign semantics, not live navigation.
"""
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .direction_geometry import detect_direction_arrows
from .ocr import read_text
from .sign_text import is_prompt_text
from .direction_groups import (associate_shared_arrow, associate_banded_rows,
                               is_destination_candidate, has_unexplained_structure)
from .panel_detector import detect_panels
from .perspective import rectify_panel, transform_points
from .ocr_refinement import refine_ocr_lines

METHOD = 'evidence_direction_rectified_v3'
DIRECTION_NAMES = {'up': '直行', 'left': '左转', 'right': '右转'}
REASONS = {
    'no_sign_detected': '未检测到大型指路牌，无法继续解析。',
    'small_sign': '检测到的牌面像素不足，无法可靠读取文字和箭头。',
    'no_text': '未读到足够清晰的中文地名候选。',
    'no_arrow': '部分牌面未识别到清晰、完整的独立箭头；不根据文字位置猜方向。',
    'prompt_sign': '含行车提示语或车道说明，超出本版支持的简单地名指路牌范围。',
    'low_text_confidence': '存在低置信度地名候选，无法可靠建立完整行布局。',
    'tilted_text': '文字仍明显倾斜，未能可靠校正牌面透视，暂不分配方向。',
    'clipped_text': '地名文字贴近检测裁剪边界，可能被截断，无法可靠给出完整地名。',
    'unsupported_layout': '地名和箭头不满足简单单列、逐行对应布局；可能是路网或多列牌面。',
    'ambiguous_pairing': '文字和箭头没有唯一的同行对应关系。',
    'unsupported_direction': '箭头方向不属于本版支持的直行、左转、右转。',
    'shared_arrow_count': '无法确认只有一个独立箭头对应这一组地名。',
    'shared_group_layout': '多个地名未形成受支持的共享箭头布局。',
    'unexplained_road_structure': '文字和箭头之外仍有未解释的大图形，可能包含复杂路网，暂不分配方向。',
    'no_verified_row_bands': '没有确认清晰的牌面分行边界。',
}


def _center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def _destination_candidate(text):
    # Deliberately conservative lexical gate. OCR text is never corrected.
    return is_destination_candidate(text)


def associate_rows(lines, arrows, size):
    """Link independently measured arrows and OCR in crop coordinates.

    Tests may inject observations here; end-to-end runs must use read_directions.
    One text line and one arrow per row only. Multiple destinations in a row,
    distance boards, connected maps and diagonal arrows are intentionally out.
    """
    reasons, relations = [], []
    if any(is_prompt_text(line['text']) for line in lines):
        return [], ['prompt_sign']
    candidates = [line for line in lines if _destination_candidate(line['text'])]
    if not candidates:
        return [], ['no_text']
    if not arrows:
        return [], ['no_arrow']
    if any(line['score'] < 0.85 or line['bbox'][3]-line['bbox'][1] < 16 for line in candidates):
        return [], ['low_text_confidence']
    for line in candidates:
        quad = line.get('quad')
        if quad and abs(math.degrees(math.atan2(quad[1][1]-quad[0][1], quad[1][0]-quad[0][0]))) > 12:
            return [], ['tilted_text']
    width, height = size
    if any(min(line['bbox'][0], line['bbox'][1], width-line['bbox'][2], height-line['bbox'][3]) <= 3
           for line in candidates):
        return [], ['clipped_text']
    if not 1 <= len(candidates) <= 5:
        return [], ['unsupported_layout']
    # A simple sign has one label column and one independent-arrow column.
    # Requiring all candidate rows prevents a lone fragment of a map being used.
    text_starts = [line['bbox'][0] for line in candidates]
    if max(text_starts)-min(text_starts) > width*0.16:
        return [], ['unsupported_layout']
    if len(arrows) > 1:
        centers = [_center(a['bbox'])[0] for a in arrows]
        if max(centers)-min(centers) > width*0.16:
            return [], ['unsupported_layout']
    matches = []
    for line in candidates:
        tb = line['bbox']; tc = _center(tb); th = tb[3]-tb[1]
        possible = []
        for arrow in arrows:
            ab = arrow['bbox']; ac = _center(ab); ah = ab[3]-ab[1]
            separated = ab[2] <= tb[0] or ab[0] >= tb[2]
            same_row = abs(tc[1]-ac[1]) <= max(th, ah)*0.55
            scale_ok = 0.4 <= ah / max(1, th) <= 3.5
            if separated and same_row and scale_ok:
                possible.append(arrow)
        if len(possible) != 1:
            return [], ['ambiguous_pairing']
        matches.append((line, possible[0]))
    ids = [a['arrow_id'] for _, a in matches]
    if len(set(ids)) != len(ids) or len(set(ids)) != len(arrows):
        return [], ['ambiguous_pairing']
    sides = {a['bbox'][0] > line['bbox'][2] for line, a in matches}
    if len(sides) != 1:
        return [], ['unsupported_layout']
    for line, arrow in matches:
        if arrow['direction'] not in DIRECTION_NAMES:
            reasons.append('unsupported_direction')
            continue
        relations.append({'destination': line['text'], 'direction': arrow['direction'],
                          'direction_label': DIRECTION_NAMES[arrow['direction']],
                          'text_id': line['text_id'], 'arrow_id': arrow['arrow_id'],
                          'text_bbox': line['bbox'], 'arrow_bbox': arrow['bbox'],
                          'ocr_score': line['score'], 'shape_iou': arrow['shape_iou'],
                          'direction_margin': arrow['direction_margin'],
                          'association_rule': 'unique_horizontal_row'})
    return relations, sorted(set(reasons))


def _offset_box(box, left, top):
    return [round(box[0]+left, 1), round(box[1]+top, 1),
            round(box[2]+left, 1), round(box[3]+top, 1)]


def _box_quad(box):
    x,y,r,b=box
    return [[x,y],[r,y],[r,b],[x,b]]


def _map_evidence(item, inverse, left, top):
    """Inverse-map ALL corners; two diagonal points are not enough under tilt."""
    def points(values):
        return [[round(x+left,1),round(y+top,1)] for x,y in transform_points(values,inverse)]
    mapped={**item}
    if 'bbox' in item:
        bounds=points(_box_quad(item['bbox']))
        mapped['bbox']=[min(p[0] for p in bounds),min(p[1] for p in bounds),
                        max(p[0] for p in bounds),max(p[1] for p in bounds)]
        mapped['quad']=points(item.get('quad') or _box_quad(item['bbox']))
        if 'height' in item:
            mapped['height']=round(mapped['bbox'][3]-mapped['bbox'][1],1)
    if 'center' in item:
        mapped['center']=points([item['center']])[0]
    return mapped


def read_directions(image, detector, conf=0.25, panel_recovery=True):
    """Return JSON evidence in EXIF-normalized original-image coordinates."""
    if image is None:
        raise ValueError('请先上传道路原图。')
    image = ImageOps.exif_transpose(image).convert('RGB')
    detection_result = (detect_panels(image,detector,conf,allow_geometry_only=False)
                        if panel_recovery else detector.detect(image, conf))
    signs = []
    for detection in detection_result['detections']:
        if detection['class_id'] != 2:
            continue
        box = detection['bbox']
        left, top = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
        right, bottom = min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3]))
        supported_quad=detection.get('geometry',{}).get('source_quad')
        if supported_quad:
            # Leave a visible margin around a measured border. A tightly
            # recovered panel box otherwise makes its own corners look clipped.
            margin=max(12,round(min(right-left,bottom-top)*.04))
            left=max(0,left-margin);top=max(0,top-margin)
            right=min(image.width,right+margin);bottom=min(image.height,bottom+margin)
        crop = image.crop((left, top, right, bottom))
        record = {'sign_id': len(signs)+1, 'detection_bbox': box,
                  'detection_confidence': detection['confidence'],
                  'crop_bbox': [left, top, right, bottom], 'texts': [], 'arrows': [],
                  'relations': [], 'status': 'abstain', 'reason_codes': []}
        record['panel_evidence']={k:detection[k] for k in
            ('proposal_source','yolo_supported','confidence_kind','seed_yolo_bbox','yolo_support','geometry')
            if k in detection}
        if min(crop.size) < 128 or crop.width*crop.height < 25600:
            record['reason_codes'] = ['small_sign']
        else:
            local_quad=([[x-left,y-top] for x,y in supported_quad]
                        if supported_quad else None)
            rectification=(rectify_panel(crop,source_quad=local_quad) if local_quad else
                           rectify_panel(crop))
            analysis=rectification['image']
            inverse=rectification['inverse_homography']
            frame={'coordinate_frame':('rectified_panel_pixels' if rectification['status']=='rectified'
                                        else 'crop_local_pixels'),
                   'origin_in_original':[left,top], 'inverse_homography_to_crop':inverse}
            record['perspective']={k:v for k,v in rectification.items() if k!='image'}
            record['perspective']['source_quad_coordinate_frame']='crop_local_pixels'
            if rectification['source_quad']:
                record['perspective']['source_quad_original']=[
                    [round(x+left,1),round(y+top,1)] for x,y in rectification['source_quad']]
            lines = read_text(analysis, min_score=0.5)['lines']
            for n, line in enumerate(lines, 1):
                line['text_id'] = f't{n}'
            refinement=refine_ocr_lines(analysis,lines)
            lines=refinement['lines']
            record['ocr_refinement_diagnostics']={**refinement['diagnostics'],**frame}
            geometry = detect_direction_arrows(analysis, lines)
            arrows = geometry['arrows']
            relations, reasons = associate_rows(lines, arrows, analysis.size)
            association={'method':'unique_horizontal_row'}
            if relations:
                structures=has_unexplained_structure(analysis,lines,arrows)
                if structures:
                    relations,reasons=[],['unexplained_road_structure']
                    association['unexplained_structures']=structures
            if not relations and len(arrows)==1:
                shared, shared_reasons, info=associate_shared_arrow(lines,arrows,analysis)
                if shared:
                    relations,reasons,association=shared,shared_reasons,info
                elif reasons[0] not in ('low_text_confidence','clipped_text','tilted_text','prompt_sign'):
                    reasons,association=shared_reasons,info
            elif not relations and len(arrows)>1:
                grouped, group_reasons, info=associate_banded_rows(lines,arrows,analysis)
                if grouped:
                    relations,reasons,association=grouped,group_reasons,info
                elif group_reasons!=['no_verified_row_bands']:
                    reasons,association=group_reasons,info
            record['association_diagnostics']={**association,**frame}
            if 'unexplained_by_band' in record['association_diagnostics']:
                record['association_diagnostics']['unexplained_by_band']=[
                    {**{k:v for k,v in band.items() if k!='origin_in_crop'},
                     'origin_in_analysis':band.get('origin_in_crop'),
                     'origin_coordinate_frame':frame['coordinate_frame']}
                    for band in record['association_diagnostics']['unexplained_by_band']]
            record['reason_codes'] = reasons
            record['arrow_diagnostics'] = {**{k:v for k,v in geometry.items() if k != 'arrows'},
                **frame}
            for line in lines:
                mapped=_map_evidence(line,inverse,left,top)
                if 'ocr_refinement' in mapped:
                    # Nested revision observations deliberately retain the analysis
                    # frame so the OCR check can be reconstructed without ambiguity.
                    mapped['ocr_refinement']={**mapped['ocr_refinement'],**frame}
                record['texts'].append(mapped)
            for arrow in arrows:
                mapped=_map_evidence(arrow,inverse,left,top)
                mapped['direction_frame']=rectification['direction_frame']
                mapped['measurement_frame']=frame['coordinate_frame']
                record['arrows'].append(mapped)
            for relation in relations:
                mapped={**relation}
                for kind in ('text','arrow'):
                    evidence=_map_evidence({'bbox':relation[kind+'_bbox']},inverse,left,top)
                    mapped[kind+'_bbox']=evidence['bbox']
                    mapped[kind+'_quad']=evidence['quad']
                mapped['direction_frame']=rectification['direction_frame']
                if 'text_segments' in mapped:
                    mapped['text_segments']=[_map_evidence(part,inverse,left,top) for part in mapped['text_segments']]
                record['relations'].append(mapped)
            record['status'] = ('partial' if reasons else 'ok') if relations else 'abstain'
        record['uncertainty_reasons'] = [REASONS[r] for r in record['reason_codes']]
        signs.append(record)
    return {'schema_version': 2, 'method': METHOD, 'experimental': True,
            'coordinate_frame': 'exif_normalized_original_pixels', 'image_size': list(image.size),
            'detector': {'entry': 'original_image_yolo', 'threshold': conf,
                         'panel_recovery': panel_recovery,
                         'diagnostics': detection_result.get('diagnostics'),
                         'all_detections': detection_result['detections'],
                         'elapsed_ms': detection_result.get('elapsed_ms')},
            'signs': signs, 'status': (('partial' if any(s['status']!='ok' for s in signs) else 'ok')
                                      if any(s['relations'] for s in signs) else 'abstain'),
            'uncertainty_reasons': [] if signs else [REASONS['no_sign_detected']],
            'scope': 'experimental independent rows and one-arrow destination groups; up/left/right only; not navigation'}


def summarize_directions(report):
    rows = ['方向解析试验版：结论表示牌面指向，需结合证据图核对。']
    if not report['signs']:
        rows.extend(report['uncertainty_reasons'])
    for sign in report['signs']:
        rows.append(f"牌 {sign['sign_id']}：")
        for relation in sign['relations']:
            rows.append(f"- {relation['destination']} — {relation['direction_label']} "
                        f"（文字 {relation['text_id']} ↔ 箭头 {relation['arrow_id']}）")
        if not sign['relations']:
            rows.append('未输出方向关系。')
        rows.extend(sign['uncertainty_reasons'])
    return '\n\n'.join(rows)


def draw_evidence(image, report, output_path):
    image = ImageOps.exif_transpose(image).convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', max(16, image.width//100))
    stroke = max(2, image.width//700)
    def label(box, text, color, quad=None):
        if quad:
            draw.line([tuple(p) for p in quad]+[tuple(quad[0])],fill=color,width=stroke)
        else:
            draw.rectangle(box, outline=color, width=stroke)
        pos = (box[0], max(0, box[1]-font.size-4))
        area = draw.textbbox(pos, text, font=font)
        draw.rectangle(area, fill='#10202c')
        draw.text(pos, text, font=font, fill=color)
    for sign in report['signs']:
        label(sign['detection_bbox'], f"牌{sign['sign_id']} {sign['status']}", '#ffcc55')
        perspective=sign.get('perspective',{})
        if perspective.get('status')=='rectified':
            quad=perspective['source_quad_original']
            draw.line([tuple(p) for p in quad]+[tuple(quad[0])],fill='#ff9955',width=stroke)
        for line in sign['texts']:
            label(line['bbox'], line['text_id'], '#4fdfff',line.get('quad'))
        for arrow in sign['arrows']:
            label(arrow['bbox'], arrow['arrow_id']+' '+arrow['direction'], '#ed9dff',arrow.get('quad'))
        for relation in sign['relations']:
            parts=relation.get('text_segments') or [{'bbox':relation['text_bbox']}]
            b=_center(relation['arrow_bbox'])
            for part in parts:
                a=_center(part['bbox'])
                draw.line((tuple(a),tuple(b)),fill='#5cff85',width=stroke)
            a=_center(parts[0]['bbox'])
            draw.text(tuple(a),relation['direction_label'],font=font,fill='#5cff85',stroke_width=1,stroke_fill='black')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)
