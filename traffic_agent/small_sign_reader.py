"""Conservative, annotation-free small-sign matching against official illustrations.

The illustrated signs are cached outside Git. A result is explanatory only when
the crop is large/clear, its color agrees with the coarse class, its template
score is strong and distinct, and the catalog contains a reviewed paraphrase.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .color_sign_proposals import color_sign_proposals
from .config import ROOT, RUNTIME
from .ocr import read_text

CATALOG = ROOT / 'knowledge' / 'small_sign_source_index.json'
CACHE = RUNTIME / 'small_sign_refs'
TARGET_CLASSES = {0: 'wran', 1: 'ban', 3: 'point-s'}
NUMERIC_NAMES = {'限制速度', '最低限速标志', '限制高度'}
TEXT_PICTOGRAM_EXCEPTIONS = {'停车让行', '减速让行'}


def _rgb(image):
    return np.asarray(ImageOps.exif_transpose(image).convert('RGB'))


def _color_share(rgb, class_id):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    if class_id == 0:
        mask = ((h >= 12) & (h <= 42) & (s >= 45) & (v >= 45))
    elif class_id == 1:
        mask = (((h <= 12) | (h >= 170)) & (s >= 45) & (v >= 45))
    else:
        mask = ((h >= 88) & (h <= 140) & (s >= 40) & (v >= 35))
    return float(mask.mean())


def _variants(rgb):
    h, w = rgb.shape[:2]
    if w / h > 1.7:
        count = round(w / h)
        if count in (2, 3):
            return [rgb[:, round(i * w / count):round((i + 1) * w / count)]
                    for i in range(count)]
    return [rgb]


def _features(rgb):
    resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (8, 8), 9).compute(gray)
    hog = hog.ravel().astype(np.float32)
    hog /= max(float(np.linalg.norm(hog)), 1e-8)
    edges = cv2.Canny(gray, 55, 140).astype(np.float32) / 255.0
    return hog, edges


def _similarity(a, b):
    ah, ae = a
    bh, be = b
    hog_score = float(np.dot(ah, bh))
    # A little spatial evidence breaks ties between signs with identical borders.
    edge_score = 1.0 - float(np.mean(np.abs(cv2.GaussianBlur(ae, (7, 7), 0) -
                                             cv2.GaussianBlur(be, (7, 7), 0))))
    return round(0.8 * hog_score + 0.2 * edge_score, 4)


def _load_catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def _references():
    """Only entries with a researched meaning are eligible to be explained."""
    CACHE.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()
    prepared = []
    for entry in catalog['entries']:
        if entry['review_state'] != 'research_paraphrase_only' or not entry['meaning_summary']:
            continue
        path = CACHE / (entry['id'] + '.png')
        if not path.is_file():
            try:
                response = requests.get(entry['source_image_url'], timeout=12,
                                        headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    continue
                with Image.open(BytesIO(response.content)) as handle:
                    handle.verify()
                path.write_bytes(response.content)
            except (requests.RequestException, OSError):
                continue
        try:
            with Image.open(path) as handle:
                rgb = _rgb(handle)
            for variant in _variants(rgb):
                prepared.append((entry, _features(variant)))
        except (OSError, ValueError, cv2.error):
            continue
    return catalog, prepared


def _number(crop, name):
    # A mild sharpening pass rescued a clear speed-60 crop whose raw OCR was empty.
    # Keep it behind the template and color checks in match_crop.
    variants = [crop]
    if name == '限制速度':
        variants.append(crop.filter(ImageFilter.UnsharpMask(radius=2, percent=180)))
    for index, variant in enumerate(variants):
        try:
            lines = read_text(variant, min_score=0.7)['lines']
        except (RuntimeError, ValueError):
            continue
        for line in lines:
            if index and line['score'] < 0.9:
                continue
            value = line['text'].replace('O', '0').replace('o', '0')
            pattern = r'(?<!\d)(\d{1,3})(?!\d)' if name != '限制高度' else r'(?<!\d)(\d{1,2}(?:\.\d)?)(?!\d)'
            match = re.search(pattern, value)
            if not match:
                continue
            parsed = float(match.group(1)) if name == '限制高度' else int(match.group(1))
            if 0 < parsed <= (10 if name == '限制高度' else 160):
                return {'value': parsed, 'unit': 'm' if name == '限制高度' else 'km/h',
                        'text': line['text'], 'ocr_score': line['score']}
    return None


def _ocr_lines(crop):
    try:
        return read_text(crop, min_score=0.85)['lines']
    except (RuntimeError, ValueError):
        return []


def _red_ring_share(rgb):
    hsv = cv2.cvtColor(cv2.resize(rgb, (128, 128)), cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    red = (((hue <= 12) | (hue >= 170)) & (sat >= 45) & (val >= 45))
    yy, xx = np.ogrid[:128, :128]
    radius = np.sqrt(((xx - 63.5) / 63.5) ** 2 + ((yy - 63.5) / 63.5) ** 2)
    annulus = (radius >= 0.58) & (radius <= 0.94)
    return float(red[annulus].mean())


def _numeric_override(crop, rgb, refs, class_id):
    ring_share = _red_ring_share(rgb)
    if class_id != 1 or ring_share < 0.15:
        return None
    lines = _ocr_lines(crop)
    if len(lines) != 1 or lines[0]['score'] < 0.9:
        return None
    center = lines[0].get('center')
    if center is None or abs(center[0] / crop.width - 0.5) > 0.20 or abs(center[1] / crop.height - 0.5) > 0.07:
        return None
    text = lines[0]['text'].replace('O', '0').replace('o', '0')
    if not re.fullmatch(r'\d{2,3}', text):
        return None
    value = int(text)
    if value not in {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}:
        return None
    entry = next((e for e, _ in refs if e['name'] == '限制速度'), None)
    if entry is None:
        return None
    feature = _features(rgb)
    speed_score = max(_similarity(feature, ref) for e, ref in refs if e['name'] == '限制速度')
    best_other = max((_similarity(feature, ref) for e, ref in refs
                      if e['dataset_class_id'] == 1 and e['name'] != '限制速度'), default=0.0)
    if speed_score < 0.64 or speed_score + 0.02 < best_other:
        return None
    return {'accepted': True, 'name': entry['name'], 'meaning': entry['meaning_summary'],
            'catalog_id': entry['id'], 'source_page': _load_catalog()['source_page'],
            'source_image_url': entry['source_image_url'],
            'number': {'value': value, 'unit': 'km/h', 'text': lines[0]['text'],
                       'ocr_score': lines[0]['score']}, 'method': '红色圆环 + 单个清晰数字',
            'scope_note': '辅助牌与具体适用范围尚未解析',
            'ring_share': round(ring_share, 3),
            'score': speed_score, 'runner_up_score': best_other,
            'margin': round(speed_score - best_other, 4)}


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def match_crop(crop, class_id, refs=None, *, min_side=96, min_score=None, min_margin=None):
    """Return evidence and a reason even for abstentions; never fabricate a name."""
    if class_id not in TARGET_CLASSES:
        return {'accepted': False, 'reason': '不属于小标志目标类别'}
    if min(crop.size) < min_side:
        return {'accepted': False, 'reason': f'牌面短边小于 {min_side} 像素'}
    ratio = crop.width / crop.height
    if not 0.72 <= ratio <= 1.38:
        return {'accepted': False, 'reason': '牌面不是清晰独立的近方形小标志'}
    rgb = _rgb(crop)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(cv2.resize(gray, (128, 128)), cv2.CV_64F).var())
    if blur < 35:
        return {'accepted': False, 'reason': '牌面模糊或对比度不足', 'sharpness': round(blur, 1)}
    color = _color_share(rgb, class_id)
    if color < 0.025:
        return {'accepted': False, 'reason': '颜色与该标志大类不一致或夜间难以分辨',
                'color_share': round(color, 3)}
    if refs is None:
        _, refs = _references()
    if not refs:
        return {'accepted': False, 'reason': '官方图示缓存不可用'}
    if (numeric := _numeric_override(crop, rgb, refs, class_id)) is not None:
        numeric.update(sharpness=round(blur, 1), color_share=round(color, 3))
        return numeric
    feature = _features(rgb)
    # Collapse alternate drawings of the same named sign before runner-up comparison.
    scores = {}
    entries = {}
    for entry, reference_feature in refs:
        if entry['dataset_class_id'] != class_id:
            continue
        key = entry['name']
        score = _similarity(feature, reference_feature)
        if score > scores.get(key, -1):
            scores[key], entries[key] = score, entry
    if not scores:
        return {'accepted': False, 'reason': '该大类暂无可用的已释义图示'}
    ordered = sorted(scores.items(), key=lambda row: -row[1])
    name, score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    if min_score is None:
        min_score = {0: 0.74, 1: 0.74, 3: 0.66}[class_id]
    if min_margin is None:
        min_margin = {0: 0.07, 1: 0.07, 3: 0.055}[class_id]
    result = {'accepted': False, 'candidate': name, 'score': score,
              'runner_up_score': runner_up, 'margin': round(score - runner_up, 4),
              'sharpness': round(blur, 1), 'color_share': round(color, 3)}
    if name == '减速让行' and score >= 0.60 and score - runner_up >= 0.10:
        center_words = [line for line in _ocr_lines(crop)
                        if line['text'] == '让' and line['score'] >= 0.95
                        and abs(line['center'][0] / crop.width - 0.5) < 0.25
                        and abs(line['center'][1] / crop.height - 0.5) < 0.25]
        if center_words:
            entry = entries[name]
            result.update(accepted=True, name=name, meaning=entry['meaning_summary'],
                          catalog_id=entry['id'], source_page=_load_catalog()['source_page'],
                          source_image_url=entry['source_image_url'], number=None,
                          method='倒三角图示 + 中央“让”字',
                          scope_note='辅助牌与具体适用范围尚未解析')
            return result
    if score < min_score or score - runner_up < min_margin:
        result['reason'] = '与官方图示不够相似，或两个候选难以区分'
        return result
    entry = entries[name]
    if name not in TEXT_PICTOGRAM_EXCEPTIONS and name not in NUMERIC_NAMES:
        if any(re.search(r'[\u3400-\u9fffA-Za-z0-9]', line['text']) and line['score'] >= 0.9
               for line in _ocr_lines(crop)):
            result['reason'] = '牌面含可读文字，图案模板不能可靠解释其完整含义'
            return result
    numeric = _number(crop, name) if name in NUMERIC_NAMES else None
    if name in NUMERIC_NAMES and numeric is None:
        result['reason'] = '牌型可能匹配，但关键数值未可靠读出'
        return result
    result.update(accepted=True, name=name, meaning=entry['meaning_summary'],
                  catalog_id=entry['id'], source_page=_load_catalog()['source_page'],
                  source_image_url=entry['source_image_url'], number=numeric,
                  scope_note='辅助牌与具体适用范围尚未解析')
    return result


def read_small_signs(image, detector, confidence=0.25, *, color_recovery=True):
    if image is None:
        raise ValueError('请先上传道路图片。')
    source = ImageOps.exif_transpose(image).convert('RGB')
    detection = detector.detect(source, conf=confidence)
    catalog, refs = _references()
    signs = []
    for i, row in enumerate(detection['detections'], 1):
        class_id = row['class_id']
        if class_id not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = row['bbox']
        box = (max(0, int(x1)), max(0, int(y1)),
               min(source.width, int(x2)), min(source.height, int(y2)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if row['confidence'] < 0.2:
            verdict = {'accepted': False, 'reason': 'YOLO 检测置信度低于 0.20'}
        else:
            verdict = match_crop(source.crop(box), class_id, refs)
        signs.append({'sign_id': i, 'class_id': class_id, 'class_name': row['class_name'],
                      'detector_source': 'yolo', 'detection_confidence': row['confidence'],
                      'bbox': list(box), **verdict})
    proposals = color_sign_proposals(source) if color_recovery else []
    reviewed = 0
    recovered = 0
    next_sign_id = max((sign['sign_id'] for sign in signs), default=0) + 1
    for proposal in proposals:
        box = proposal['bbox']
        class_id = proposal['class_id']
        if min(box[2]-box[0], box[3]-box[1]) < 96:
            continue
        if any(sign['class_id'] == class_id and _box_iou(sign['bbox'], box) >= 0.5
               for sign in signs):
            continue
        reviewed += 1
        verdict = match_crop(source.crop(tuple(box)), class_id, refs)
        if not verdict['accepted']:
            continue
        if any(sign['accepted'] and _box_iou(sign['bbox'], box) >= 0.5 for sign in signs):
            continue
        signs.append({'sign_id': next_sign_id, 'class_id': class_id,
                      'class_name': TARGET_CLASSES[class_id],
                      'detector_source': 'color_shape', 'detection_confidence': None,
                      'proposal_color_fraction': proposal['color_fraction'],
                      'bbox': list(box), **verdict})
        recovered += 1
        next_sign_id += 1
    return {'signs': signs, 'accepted': sum(s['accepted'] for s in signs),
            'abstained': sum(not s['accepted'] for s in signs),
            'color_proposals': len(proposals), 'color_proposals_reviewed': reviewed,
            'color_proposals_accepted': recovered,
            'references_loaded': len(refs), 'catalog_source': catalog['source_page'],
            'method': 'YOLO 检测 + 颜色/形状候选补充，经图示和 OCR 复核；试验版，未经冻结评测'}


def draw_small_sign_evidence(image, report, output_path):
    canvas = ImageOps.exif_transpose(image).convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(15, canvas.width // 95))
    for sign in report['signs']:
        color = '#17a34a' if sign['accepted'] else '#f59e0b'
        draw.rectangle(sign['bbox'], outline=color, width=max(3, canvas.width // 500))
        origin = 'CV' if sign.get('detector_source') == 'color_shape' else 'Y'
        label = f"#{sign['sign_id']} {origin} {'MATCH' if sign['accepted'] else 'UNSURE'}"
        x, y = sign['bbox'][:2]
        draw.text((x, max(0, y - 20)), label, fill=color, font=font,
                  stroke_width=2, stroke_fill='black')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def summarize_small_signs(report):
    if not report['signs']:
        reviewed = report.get('color_proposals_reviewed', 0)
        return (f'未检测到可解读的 0/1/3 类小标志；颜色/形状候选复核 {reviewed} 个，'
                '均未通过证据检查。这不代表图片里不存在标志。')
    lines = [f"检测到 {len(report['signs'])} 个 0/1/3 类框；具体含义通过 {report['accepted']} 个，拒答 {report['abstained']} 个。"]
    if report.get('color_proposals_reviewed'):
        lines.append(f"另复核颜色/形状候选 {report['color_proposals_reviewed']} 个，"
                     f"新增通过 {report['color_proposals_accepted']} 个；未通过者不作为具体标志输出。")
    for sign in report['signs']:
        if sign['accepted']:
            suffix = ''
            if sign.get('number'):
                number = sign['number']
                suffix = f"牌面数字 {number['value']} {number['unit']}（OCR {number['ocr_score']:.2f}）。"
            source = '颜色/形状候选' if sign.get('detector_source') == 'color_shape' else 'YOLO'
            lines.append(f"- #{sign['sign_id']}（{source}） **{sign['name']}**：{sign['meaning']} {suffix} "
                         f"[图解来源]({sign['source_page']})。{sign['scope_note']}。")
        else:
            lines.append(f"- #{sign['sign_id']} 不解读：{sign['reason']}。")
    lines.append('仅作图像理解试验；未经过独立人工标注的冻结测试。')
    return '\n'.join(lines)
