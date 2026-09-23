"""牌面箭头检测：在蓝底指路牌上定位白色箭头并判定方向。

方法（无需额外训练，先建立可评估的基线）：
1. 牌面按亮度/饱和度阈值提取白色前景（箭头、文字、边框、线路示意都是白色）；
2. 连通域分析，用面积、外接框比、实心度、端点距离过滤掉文字与细线；
3. 用 PCA 主轴 + 尖端判定得到箭头指向。

局限：当牌面使用“路网示意图”而非独立箭头时（如图例 08117），本模块不会给出方向，
而应由上层标记为不确定，不得猜测。
"""

from functools import lru_cache
from math import hypot
from pathlib import Path

import cv2
import numpy as np

DIRECTIONS = ('left', 'slight_left', 'up', 'slight_right', 'right', 'down')
# 白/浅色前景阈值（BGR 空间下按 HSV 取值）
WHITE_MAX_SAT = 90
WHITE_MIN_VAL = 150
# 模板匹配参数
TEMPLATE_SIZES = (0.05, 0.07, 0.09, 0.12, 0.16, 0.22)
MATCH_THRESHOLD = 0.55
NMS_IOU = 0.30


def white_mask(bgr):
    """提取“浅色前景”（文字与箭头）。

    实测结论（08342_2）：夜间/偏暗照片里牌面白字是**蓝白**（B≫G≈R，饱和度甚至高于蓝底），
    因此按亮度或饱和度做全局阈值都会失败。可用判据是**局部对比度**（字比紧邻背景亮）。
    这里用局部自适应阈值，并在候选里选“前景占比更接近典型牌面”的那个。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = max(31, (min(gray.shape[:2])//10) | 1)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, block, -6)
    kernel = np.ones((3, 3), np.uint8)
    candidates = [otsu, adaptive]
    ratio_of = lambda mask: float((mask > 0).mean())   # noqa: E731
    typical = 0.22
    chosen = min(candidates, key=lambda mask: abs(ratio_of(mask)-typical))
    return cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, kernel, iterations=1)


def mask_boxes(mask, boxes, pad=0.0):
    """把 OCR 检出的文字区域从白色前景里抹掉，避免汉字笔画被当成箭头。"""
    out = mask.copy()
    for box in boxes:
        if isinstance(box, dict):
            box = box.get('bbox')
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        if pad:
            grow_x, grow_y = (x1-x0)*pad, (y1-y0)*pad
            x0, y0, x1, y1 = x0-grow_x, y0-grow_y, x1+grow_x, y1+grow_y
        cv2.rectangle(out, (int(max(0, x0)), int(max(0, y0))),
                      (int(x1), int(y1)), 0, -1)
    return out


def detect_arrows_white(image, text_boxes=(), min_area=None, max_fill=0.80,
                        min_solidity=0.62, max_area_ratio=0.08, min_side=24):
    """在抹掉文字区域后的浅色前景上找**箭头候选**（不判定方向）。

    实测（08342_1）：真箭头是面积约 1e4 像素、fill≈0.5、solidity≈0.7 的实心块；
    而牌面外框/线路示意图是 fill<0.1 或 solidity<0.3 的细长结构，文字笔画则被 OCR 遮罩清除。
    方向判定交给 sign_parser 的位置规则——对比发现形似度分类器（PCA 尖端法）
    对真箭头也只给出 0.15–0.30 的置信度，不如位置判据可靠。
    """
    bgr = _load_bgr(image)
    mask = white_mask(bgr)
    height, width = mask.shape[:2]
    if min_area is None:
        min_area = max(240, int(height*width*0.0012))
    cleaned = mask_boxes(mask, text_boxes, pad=0.06)
    count, labels, stats = _components(cleaned)
    arrows, rejected = [], {}
    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1
    for index in range(1, count):
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        w, h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or w < min_side or h < min_side:
            reject('small')
            continue
        if area > height*width*max_area_ratio:
            reject('too_large')
            continue
        component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        fill = area/float(w*h)
        if fill > max_fill:
            reject('solid')
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        hull_area = cv2.contourArea(cv2.convexHull(contour)) if contour is not None else 0
        solidity = (cv2.contourArea(contour)/hull_area) if hull_area > 0 else 0
        if solidity < min_solidity:
            reject('not_solid')
            continue
        arrows.append({'bbox':[int(x), int(y), int(x+w), int(y+h)],
                       'center':[int(x+w/2), int(y+h/2)], 'area':area,
                       'width':int(w), 'height':int(h),
                       'fill':round(fill, 3), 'solidity':round(solidity, 3)})
    arrows.sort(key=lambda item: -item['area'])
    return {'arrows':arrows, 'method':'white_after_text_mask',
            'white_ratio':round(float((mask > 0).mean()), 4),
            'text_boxes_masked':len(list(text_boxes)), 'rejected':rejected}


def _components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return count, labels, stats


def _arrow_outline(size, head=0.45, shaft=0.22, aspect=1.7):
    """生成细长型箭头模板（白底黑箭头，便于与白色前景做匹配）。"""
    height = int(size*aspect)
    width = int(size)
    canvas = np.zeros((height, width), np.uint8)
    head_h = int(height*head)
    shaft_w = max(3, int(width*shaft))
    shaft_x0 = (width-shaft_w)//2
    cv2.rectangle(canvas, (shaft_x0, head_h), (shaft_x0+shaft_w, height-1), 255, -1)
    cv2.fillPoly(canvas, [np.array([[0, head_h], [width-1, head_h], [width//2, 0]], np.int32)], 255)
    return canvas


def _rotate(image, angle):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(height*sin + width*cos), int(height*cos + width*sin)
    matrix[0, 2] += new_w/2 - center[0]
    matrix[1, 2] += new_h/2 - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST)


@lru_cache(maxsize=64)
def _templates(size, variant):
    """返回 [(direction, template_gray)]；variant 控制箭头粗细比例。"""
    head, shaft = ((0.45, 0.22), (0.38, 0.30))[variant]
    base = _arrow_outline(size, head, shaft)
    rows = []
    for direction, angle in (('up', 0), ('right', 90), ('down', 180), ('left', 270),
                             ('slight_right', 45), ('slight_left', -45)):
        rows.append((direction, _rotate(base, angle)))
    return rows


def _nms(candidates, iou_threshold=NMS_IOU):
    kept = []
    for candidate in sorted(candidates, key=lambda item: -item['score']):
        box = candidate['bbox']
        if any(_iou(box, other['bbox']) > iou_threshold for other in kept):
            continue
        kept.append(candidate)
    return kept


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right-left)*(bottom-top)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter/float(area_a + area_b - inter)


def detect_arrows_template(image, threshold=MATCH_THRESHOLD, sizes=TEMPLATE_SIZES):
    """多尺度模板匹配检测箭头；比连通域几何法更能排除汉字笔画。"""
    bgr = _load_bgr(image)
    mask = white_mask(bgr)
    foreground = mask.astype(np.float32)/255.0
    shorter = min(bgr.shape[:2])
    candidates = []
    for fraction in sizes:
        size = max(9, int(shorter*fraction))
        for variant in (0, 1):
            for direction, template in _templates(size, variant):
                if template.shape[0] > foreground.shape[0] or template.shape[1] > foreground.shape[1]:
                    continue
                response = cv2.matchTemplate(foreground, template.astype(np.float32)/255.0,
                                             cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(response >= threshold)
                for y, x in zip(ys, xs):
                    score = float(response[y, x])
                    candidates.append({'direction':direction, 'score':round(score, 3),
                                       'template_size':size, 'variant':variant,
                                       'bbox':[int(x), int(y), int(x+template.shape[1]),
                                               int(y+template.shape[0])],
                                       'center':[int(x+template.shape[1]/2),
                                                 int(y+template.shape[0]/2)]})
    arrows = []
    for candidate in _nms(candidates):
        center = candidate['center']
        arrows.append({**candidate, 'confidence':candidate['score'],
                       'inside_sign':bool(mask[max(0, center[1]-2):center[1]+2,
                                              max(0, center[0]-2):center[0]+2].mean() > 100)})
    arrows = [arrow for arrow in arrows if arrow.pop('inside_sign')]
    arrows.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    return {'arrows':arrows, 'white_ratio':round(float((mask > 0).mean()), 4),
            'method':'template', 'candidates':len(candidates)}


def _load_bgr(image):
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f'无法读取图片：{image}')
        return bgr
    return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)


def detect_arrows(image, min_score=0.05):
    """连通域几何法（基线对照用）。牌面为路网示意图时不应据此推断方向。"""
    bgr = _load_bgr(image)
    mask = white_mask(bgr)
    count, labels, stats = _components(mask)
    arrows = []
    for index in range(1, count):
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        w, h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        ok, info = _arrow_score(component, stats[index])
        if not ok:
            continue
        direction, confidence = _direction(component)
        if direction == 'unknown' or confidence < min_score:
            continue
        arrows.append({'direction':direction, 'confidence':confidence,
                       'bbox':[int(x), int(y), int(x+w), int(y+h)],
                       'center':[int(x+w/2), int(y+h/2)], 'area':int(info['area']),
                       'solidity':info.get('solidity'), 'fill':info.get('fill')})
    arrows.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    return {'arrows':arrows, 'white_ratio':round(float((mask > 0).mean()), 4),
            'method':'geometric', 'candidates':len(arrows)}


def _arrow_score(component, stats):
    """判断一个白色连通域像不像箭头，返回 (是否, 置信度线索)。"""
    height, width = component.shape
    area = int(stats[cv2.CC_STAT_AREA])
    if area < 300 or height < 18 or width < 18:
        return False, {'reason':'too_small', 'area':area}
    ratio = area/float(height*width)
    if ratio < 0.28:           # 细线、边框、路线示意
        return False, {'reason':'thin', 'fill':round(ratio, 3)}
    if ratio > 0.92:           # 实心块
        return False, {'reason':'solid', 'fill':round(ratio, 3)}
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, {'reason':'no_contour'}
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False, {'reason':'degenerate'}
    solidity = cv2.contourArea(contour)/hull_area
    if solidity < 0.62:        # 箭头是凸的，文字与示意图往往更碎
        return False, {'reason':'not_convex', 'solidity':round(solidity, 3)}
    return True, {'area':area, 'fill':round(ratio, 3), 'solidity':round(solidity, 3)}


def _direction(component):
    """用主轴方向 + 尖端位置判定箭头指向。"""
    ys, xs = np.nonzero(component)
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    projections = centered @ axis
    low, high = projections.min(), projections.max()
    mask_low = projections <= low + (high-low)*0.15
    mask_high = projections >= high - (high-low)*0.15
    width_low = float(np.linalg.norm(centered[mask_low], axis=1).std()) if mask_low.any() else 0.0
    width_high = float(np.linalg.norm(centered[mask_high], axis=1).std()) if mask_high.any() else 0.0
    tip = 'high' if width_low > width_high else 'low'   # 尖端更窄
    sign = 1.0 if tip == 'high' else -1.0
    dx, dy = axis[0]*sign, axis[1]*sign
    if abs(dy) < 0.001 and abs(dx) < 0.001:
        return 'unknown', 0.0
    length = hypot(dx, dy)
    confidence = abs(width_low-width_high)/max(width_low, width_high, 1e-6)
    if abs(dx)/length > 0.94:
        direction = 'right' if dx > 0 else 'left'
    elif abs(dy)/length > 0.94:
        direction = 'down' if dy > 0 else 'up'
    elif dx > 0:
        direction = 'slight_right'
    else:
        direction = 'slight_left'
    return direction, round(min(1.0, confidence), 3)


def detect_arrows(image, min_score=0.05):
    """在牌面图上检测箭头。image 为路径或 PIL.Image；返回 {'arrows':[...], 'mask_ratio':..}"""
    if isinstance(image, (str,)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f'无法读取图片：{image}')
    else:
        rgb = np.array(image.convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, WHITE_MIN_VAL), (180, WHITE_MAX_SAT, 255))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats = _components(mask)
    arrows = []
    for index in range(1, count):
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        w, h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        ok, info = _arrow_score(component, stats[index])
        if not ok:
            continue
        direction, confidence = _direction(component)
        if direction == 'unknown' or confidence < min_score:
            continue
        arrows.append({'direction':direction, 'confidence':confidence,
                       'bbox':[int(x), int(y), int(x+w), int(y+h)],
                       'center':[int(x+w/2), int(y+h/2)], 'area':int(info['area']),
                       'solidity':info.get('solidity'), 'fill':info.get('fill')})
    arrows.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    return {'arrows':arrows, 'white_ratio':round(float((mask > 0).mean()), 4)}
