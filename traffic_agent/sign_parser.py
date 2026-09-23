"""指路牌解析：把牌面拆成“目的地组 + 箭头”，并给出可信或不确定的对应关系。

方法（当前基线，无需额外训练）：
1. OCR 给出文字与四点坐标（traffic_agent/ocr.py）；
2. 按垂直重叠与水平间距把文字行聚成“目的地组”（同一目的地的大字/拼音/括号辅路名会成组）；
3. 在抹掉文字区域后的浅色前景上找箭头候选（traffic_agent/arrows.py）；
4. 箭头方向用**相对文字组的位置**判定：位于文字组左侧→left，右侧→right，下方→up，上方→down，
   同时结合候选自身长宽比做修正。这比只看箭头形状更稳，但仍然是启发式；
5. 置信度不足、或牌面为路网示意图（无独立箭头）时，进入 uncertain，**不猜测方向**。

输出描述的是“牌面指示信息”，不等于结合车辆位置生成的导航指令。
"""

from .arrows import detect_arrows_white
from .ocr import read_text

DIRECTION_LABELS = {'left':'左', 'slight_left':'左前', 'up':'直行',
                    'slight_right':'右前', 'right':'右', 'down':'掉头/向下'}
# 位置推断时的水平/垂直容差（相对文字组尺寸）
PAD = 0.35


def group_text(lines, overlap=0.55, gap_ratio=1.6, pad_ratio=1.6):
    """把文字行聚成目的地组：先按垂直重叠，再按水平相邻与行高相近合并。"""
    rows = sorted(lines, key=lambda line: (line['bbox'][1], line['bbox'][0]))
    groups = []
    for line in rows:
        top, bottom = line['bbox'][1], line['bbox'][3]
        height = max(1.0, bottom-top)
        placed = False
        for group in groups:
            g_top, g_bottom = group['bbox'][1], group['bbox'][3]
            inter = min(bottom, g_bottom) - max(top, g_top)
            if inter/max(1.0, min(height, g_bottom-g_top)) >= overlap:
                _extend(group, line)
                placed = True
                break
        if not placed:
            for group in groups:
                g_height = max(1.0, group['bbox'][3]-group['bbox'][1])
                if abs(height-g_height)/max(height, g_height) > 0.8:
                    continue
                gap = max(line['bbox'][0]-group['bbox'][2], group['bbox'][0]-line['bbox'][2])
                if 0 <= gap <= gap_ratio*max(height, g_height):
                    _extend(group, line)
                    placed = True
                    break
        if not placed:
            groups.append({'bbox':list(line['bbox']), 'lines':[line]})
    for group in groups:
        group['lines'].sort(key=lambda line: line['bbox'][1])
        group['text'] = ' '.join(line['text'] for line in group['lines'])
        group['center'] = [(group['bbox'][0]+group['bbox'][2])/2, (group['bbox'][1]+group['bbox'][3])/2]
        group['height'] = group['bbox'][3]-group['bbox'][1]
    groups.sort(key=lambda group: (group['bbox'][1], group['bbox'][0]))
    return groups


def _extend(group, line):
    x0 = min(group['bbox'][0], line['bbox'][0])
    y0 = min(group['bbox'][1], line['bbox'][1])
    x1 = max(group['bbox'][2], line['bbox'][2])
    y1 = max(group['bbox'][3], line['bbox'][3])
    group['bbox'] = [x0, y0, x1, y1]
    group['lines'].append(line)


def infer_direction(arrow, group):
    """按箭头相对文字组的位置推断方向；返回 (方向, 依据说明, 置信度)。"""
    ax0, ay0, ax1, ay1 = arrow['bbox']
    acx, acy = arrow['center']
    gx0, gy0, gx1, gy1 = group['bbox']
    width = max(1, ax1-ax0)
    height = max(1, ay1-ay0)
    horizontal = width >= height
    pad_x = (gx1-gx0)*PAD
    pad_y = (gy1-gy0)*PAD
    if ay0 >= gy1 - pad_y:                       # 在文字组下方
        return ('up', '位于文字组下方', 0.55) if horizontal else ('up', '位于文字组下方', 0.6)
    if ay1 <= gy0 + pad_y:                       # 在文字组上方
        return 'down', '位于文字组上方', 0.5
    if ax1 <= gx0 + pad_x:                       # 在文字组左侧
        return 'left', '位于文字组左侧', 0.7
    if ax0 >= gx1 - pad_x:                       # 在文字组右侧
        return 'right', '位于文字组右侧', 0.7
    return None, '与文字组重叠，无法确定方向', 0.0


def neighborhood_arrow(arrow_candidates, group, pad_ratio=0.6):
    """在文字组的左/右/下邻域里找与该组尺度匹配的箭头候选。

    实测（evaluate_sign_parser.py）表明：全牌面撒网会产生大量误报（歧义牌也被断言方向）。
    收窄到邻域 + 尺度匹配后，只有“紧贴某个地名组、且尺寸与该组文字高度相当”的候选才被采用。
    """
    gx0, gy0, gx1, gy1 = group['bbox']
    group_height = max(1, gy1-gy0)
    group_width = max(1, gx1-gx0)
    best = None
    for arrow in arrow_candidates:
        ax0, ay0, ax1, ay1 = arrow['bbox']
        a_height = max(1, ay1-ay0)
        # 尺度匹配：箭头高度与文字组高度同量级
        if not (0.35*group_height <= a_height <= 1.6*group_height):
            continue
        # 垂直重叠要求：箭头与该组大致同一水平带（左侧/右侧），或位于其正下方
        overlap = min(ay1, gy1) - max(ay0, gy0)
        same_band = overlap > 0.4*min(a_height, group_height)
        below = ay0 >= gy1 - 0.25*group_height
        if not (same_band or below):
            continue
        margin = group_height*pad_ratio
        gap = None
        zone = None
        if ax1 <= gx0 + 0.15*group_width:
            gap, zone = gx0-ax1, 'left'
        elif ax0 >= gx1 - 0.15*group_width:
            gap, zone = ax0-gx1, 'right'
        elif below:
            gap, zone = ay0-gy1, 'below'
        if zone is None or gap < 0 or gap > max(margin, 0.6*group_width):
            continue
        score = gap/group_height
        if best is None or score < best[0]:
            best = (score, arrow, zone)
    return best


def link(arrows, groups):
    """按邻域搜索把箭头匹配到目的地组；找不到可靠邻域箭头的组保持未关联。"""
    pairs = []
    for group in groups:
        found = neighborhood_arrow(arrows, group)
        pairs.append({'group':group, 'arrow':found[1] if found else None,
                      'zone':found[2] if found else None,
                      'gap':round(found[1] and (found[0]), 3) if found else None})
    return pairs


def infer_direction(arrow, group, zone=None):
    """按箭头所在邻域推断方向；无法确定时返回 None，交由上层标记不确定。"""
    if zone == 'left':
        return 'left', '位于文字组左侧邻域', 0.7
    if zone == 'right':
        return 'right', '位于文字组右侧邻域', 0.7
    if zone == 'below':
        ax0, ay0, ax1, ay1 = arrow['bbox']
        if (ax1-ax0) > 1.3*(ay1-ay0):
            return None, '下方候选为横向结构，方向不明', 0.0
        return 'up', '位于文字组正下方', 0.6
    return None, '文字组邻域内没有尺度匹配的箭头', 0.0


def parse_sign(image, min_score=0.5):
    """解析一块牌面，返回结构化结果；无法确定的关系进入 uncertain。"""
    text = read_text(image, min_score=min_score)
    lines = text['lines']
    groups = group_text(lines)
    detection = detect_arrows_white(image, [line['bbox'] for line in lines])
    arrows = detection['arrows']
    routes, uncertain = [], []
    for pair in link(arrows, groups):
        group = pair['group']
        if pair['arrow'] is None:
            uncertain.append({'destination':group['text'], 'destination_bbox':group['bbox'],
                              'reason':'邻域内没有尺度匹配的箭头'})
            continue
        direction, basis, confidence = infer_direction(pair['arrow'], group, pair['zone'])
        record = {'destination':group['text'], 'destination_bbox':group['bbox'],
                  'arrow_bbox':pair['arrow']['bbox'],
                  'arrow_id':f"a{len(routes)+len(uncertain)+1}",
                  'zone':pair['zone'], 'gap':pair['gap']}
        if direction is None or confidence < 0.5:
            uncertain.append({**record, 'reason':basis})
            continue
        routes.append({**record, 'direction':direction, 'direction_label':DIRECTION_LABELS[direction],
                       'confidence':confidence, 'basis':basis})
    unassigned = [group for group in groups
                  if not any(route['destination_bbox'] == group['bbox'] for route in routes)]
    return {'text_count':len(lines), 'groups':[{'text':group['text'], 'bbox':group['bbox']}
                                               for group in groups],
            'arrow_count':len(arrows), 'arrows':arrows, 'routes':routes,
            'uncertain':uncertain,
            'groups_without_arrow':[group['text'] for group in unassigned],
            'detection_rejected':detection['rejected'],
            'method':'ocr_groups + neighborhood_arrow + position_rule'}
