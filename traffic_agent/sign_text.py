"""牌面文字提取与意图推断。

针对 point-l（class 2）中包含大量**非地名类提示牌**的现实：不再强行解析“地名↔箭头”，
而是只做两件事：
1. 读出牌面上的全部文字（逐段保留 OCR 原文、置信度、四点坐标）；
2. 由语言模型根据这些文字**推断**这块牌想表达什么。

纪律：
- 文字部分只输出 OCR 结果，不做“纠正性改写”；低置信度文字单独列出；
- 推断部分必须标注为推测，并附带依据（引用了哪几段文字）；
- 不输出行车指令、不做安全判断。
"""

from .config import WEIGHTS
from .ocr import read_text

POINT_L = 2
DESTINATION_MIN_HEIGHT = 80
DESTINATION_MIN_LENGTH = 2
# 可读性门槛：实测夜间样例中检出的牌面裁剪仅 75x118 / 144x274 像素，肉眼都无法辨认。
# 门槛按“文字是否还能读出”定：144px 的提示牌仍可识别（08325 各行其道），
# 而 61~67px 的牌面（08274 / 08086）OCR 已无输出，故取 128px。
MIN_CROP_SIDE = 128
MIN_CROP_AREA = 128*160
# 常见“非地名”用词，用于区分提示语与地点名。
# 扩充依据：scripts/collect_prompt_signs.py 对 783 张牌面抽样 OCR 后人工核对
# （报告 outputs/prompt_sign_candidates.json、prompt_holdout.json）。
#
# 两个词表的分工（留出集暴露的问题）：
# - STRONG_PROMPT_WORDS：本身就是行车要求，单独出现即可判定为提示语；
# - WEAK_PROMPT_WORDS：泛指词（车辆、行车、车道…），单独出现不足以判定——
#   实测“车辆管理分所”“公路自行车赛场”会被误判，故必须与强标记同现才计入。
STRONG_PROMPT_WORDS = ('请按', '请选', '请进', '请止', '请系', '严禁', '禁止', '各行其道',
                       '文明', '礼让', '减速', '慢行', '车距', '疲劳', '违者', '后果自负',
                       '施工', '限速', '测速', '抓拍', '区间测速', '事故多发', '安全带')
WEAK_PROMPT_WORDS = ('车道', '车辆', '行驶', '行车', '借用', '导向', '集散', '辅道',
                     '停车', '注意', '保持', '监控', '警告', '安全')
# 设施/服务信息：告知停车场、场馆、单位等，不是对驾驶员的行为要求。
INFO_WORDS = ('停车场', '停车库', '泊车', '服务区', '收费站', '加油站', '充电站',
              '体育馆', '体育场', '赛场', '纪念馆', '博物馆', '游客中心', '管理分所')
PLACE_SUFFIX = ('路', '街', '桥', '大道', '巷', '镇', '村', '市', '区', '县', '乡', '站',
                '机场', '高速', '公路', '广场', '港', '口', '门', '塔', '园')


def is_place_name(text):
    return any(text.endswith(suffix) for suffix in PLACE_SUFFIX)


def is_prompt_text(text):
    """是否可判定为行车规则/提示语。

    强标记单独出现即可；泛指词必须与强标记同现（否则“车辆管理分所”“公路自行车赛场”会误判）。
    """
    strong = [word for word in STRONG_PROMPT_WORDS if word in text]
    if strong:
        return True
    if '请' in text:
        return True
    weak = [word for word in WEAK_PROMPT_WORDS if word in text]
    return len(weak) >= 2


def classify_text(text):
    """把一段牌面文字分类；提示/规则类优先于地名，因为“前方路口禁止左转”同时含两者。

    返回 (类型, 命中的关键词)；类型取值：prompt / info / place / code / other。
    - prompt：对驾驶员的行为要求（请按导向车道行驶、减速慢行、禁止通行…）；
    - info：设施或服务告知（停车场、场馆、单位名…），不是行为要求；
    - place：地名或道路名；code：编号（G5513/S230）；other：置信度不足或无法归类。
    """
    strong = [word for word in STRONG_PROMPT_WORDS if word in text]
    if strong:
        return 'prompt', strong
    if '请' in text:
        return 'prompt', ['请']
    weak = [word for word in WEAK_PROMPT_WORDS if word in text]
    if len(weak) >= 2:
        return 'prompt', weak
    info = [word for word in INFO_WORDS if word in text]
    if info:
        return 'info', info
    if text.strip() in ('P', 'p'):
        return 'code', ['停车标志']
    if is_place_name(text):
        return 'place', [text[-1]]
    return 'other', []


def split_lines(lines):
    """把 OCR 行粗分为“目的地名 / 辅助信息 / 提示语 / 低置信度”。"""
    buckets = {'destinations':[], 'supporting':[], 'prompts':[], 'low_confidence':[]}
    for line in lines:
        text = line['text']
        if line['score'] < 0.7:
            buckets['low_confidence'].append(line)
            continue
        kind, _ = classify_text(text)
        if kind == 'prompt':
            buckets['prompts'].append(line)
            continue
        ascii_only = all(ord(ch) < 128 for ch in text)
        if kind == 'place' and len(text) >= DESTINATION_MIN_LENGTH:
            buckets['destinations'].append(line)
        elif not ascii_only and len(text) >= DESTINATION_MIN_LENGTH and line.get('height',
                                                                               0) >= DESTINATION_MIN_HEIGHT:
            labels = ('（', '(', '）', ')')
            if text.startswith(labels) or text.endswith(labels):
                buckets['supporting'].append(line)
            else:
                buckets['destinations'].append(line)
        else:
            buckets['supporting'].append(line)
    return buckets


def _detect_point_l(image, detector, conf):
    detections = detector.detect(image, conf)['detections']
    return [item for item in detections if item['class_id'] == POINT_L]


def crop_for(image, detection, padding=0.04):
    box = detection['bbox']
    width, height = image.size
    pad_x, pad_y = (box[2]-box[0])*padding, (box[3]-box[1])*padding
    left = int(max(0, box[0]-pad_x))
    top = int(max(0, box[1]-pad_y))
    right = int(min(width, box[2]+pad_x))
    bottom = int(min(height, box[3]+pad_y))
    return image.crop((left, top, right, bottom))


def readability(crop):
    """判断牌面裁剪是否具备可读条件；返回 (是否可读, 原因)。"""
    width, height = crop.size
    area = width*height
    if min(width, height) < MIN_CROP_SIDE or area < MIN_CROP_AREA:
        return False, (f'牌面裁剪仅 {width}x{height} 像素，低于可读门槛 '
                       f'{MIN_CROP_SIDE}px/{MIN_CROP_AREA}px²，OCR 不具备可读条件')
    return True, None


def read_signs(image, detector, conf=0.25, min_score=0.5):
    """检测 point-l 并按原图分辨率裁剪、逐块判断可读性并 OCR。"""
    signs = []
    for index, detection in enumerate(_detect_point_l(image, detector, conf), start=1):
        crop = crop_for(image, detection)
        readable, reason = readability(crop)
        record = {'sign_id':index, 'detection_bbox':[round(v, 1) for v in detection['bbox']],
                  'detection_confidence':detection['confidence'],
                  'crop_size':list(crop.size), 'readable':readable, 'readable_reason':reason,
                  'text_count':0, 'texts':[], 'lines':[],
                  'destinations':[], 'supporting':[], 'prompts':[], 'low_confidence':[],
                  'crop':crop}
        if readable:
            ocr = read_text(crop, min_score=min_score)
            buckets = split_lines(ocr['lines'])
            record.update({'text_count':len(ocr['lines']),
                           'texts':[line['text'] for line in ocr['lines']],
                           'lines':ocr['lines'],
                           'destinations':[line['text'] for line in buckets['destinations']],
                           'supporting':[line['text'] for line in buckets['supporting']],
                           'prompts':[line['text'] for line in buckets['prompts']],
                           'low_confidence':[{'text':line['text'], 'score':line['score']}
                                             for line in buckets['low_confidence']]})
        signs.append(record)
    return signs


def summarize(signs):
    """给模型与界面看的最小结构化摘要；只含 OCR 文本，不含推断结论。"""
    rows = []
    for sign in signs:
        rows.append({'sign_id':sign['sign_id'],
                     '检测置信度':round(sign['detection_confidence'], 3),
                     '可读':sign['readable'],
                     '不可读原因':sign['readable_reason'],
                     '牌面文字':sign['texts'],
                     '疑似提示语':sign['prompts'],
                     '地名候选':sign['destinations'],
                     '低置信度文字':[item['text'] for item in sign['low_confidence']]})
    return rows


def inference_prompt(signs):
    """生成给语言模型的提示：要求区分“牌面事实”与“推测含义”。"""
    lines = ['以下是从一张道路图片中裁出的指路标志牌面 OCR 文字（可能有多块牌）。',
             '请完成两件事：',
             '1. 逐块复述牌面文字（只写识别到的内容，不要改写成更通顺的句子）；',
             '2. 推测这块牌想表达什么（例如地名指示、行车规范提示、施工或限速提醒、'
             '单位/景区提示等），并说明你依据的是哪几段文字。',
             '要求：推测必须标明是推测；低置信度文字要指出来；不要给出驾驶指令或安全结论；'
             '标为“可读=false”的牌面不具备可读条件，必须直接说明无法识别，'
             '不得根据检测类别或外观猜测内容；文字太碎无法判断意图时也直接说不确定。']
    for row in summarize(signs):
        lines.append('---')
        if not row['可读']:
            lines.append(f"牌 {row['sign_id']}（检测置信度 {row['检测置信度']}）：不可读——"
                         f"{row['不可读原因']}")
            continue
        lines.append(f"牌 {row['sign_id']}（检测置信度 {row['检测置信度']}）文字："
                     + ' / '.join(row['牌面文字']))
        if row['低置信度文字']:
            lines.append('其中低置信度文字：' + ' / '.join(row['低置信度文字']))
    return '\n'.join(lines)


def load_detector(weight_path=WEIGHTS, device='cpu'):
    """延迟导入，避免仅用 OCR 时也加载推理框架。"""
    from .detector import TrafficSignDetector
    return TrafficSignDetector(weight_path, device)
