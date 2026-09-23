"""牌面文字识别（OCR）：返回文字、置信度与四点坐标。

设计要点：
- OCR 只对**原图分辨率裁出的牌面**运行，不对缩放到 640 的推理图运行；
- 保留四点旋转框坐标，而不是只留字符串：后续“地名↔箭头”关系解析依赖位置；
- 引擎延迟加载，未安装依赖时给出明确报错而不是静默降级；
- 本模块只输出牌面上“写了什么”，不输出“意味着什么”。

引擎选型：RapidOCR（onnxruntime），不依赖 PaddlePaddle，便于与本机 torch 2.3.0 共存。
注意：onnxruntime 1.30.0 在本机导入即崩（0xC0000005），已验证 1.18.1 + protobuf 4.25.3 可用。
"""

from functools import lru_cache
from pathlib import Path
from threading import Lock

_INFERENCE_LOCK = Lock()

INSTALL_HINT = ('缺少 OCR 依赖。请安装：pip install rapidocr_onnxruntime onnxruntime==1.18.1 '
                'protobuf==4.25.3')


@lru_cache(maxsize=1)
def engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc
    return RapidOCR()


def read_text(image, min_score=0.5):
    """识别图片中的文字，返回按版面顺序（先上后下、先左后右）排列的结果。

    image 可以是路径或 PIL.Image。返回：
        {'lines':[{'text','score','quad','bbox','center','height'}], 'elapsed_ms'}
    """
    if isinstance(image, (str, Path)):
        source = str(image)
        width = height = None
    else:
        # RapidOCR interprets ndarray as BGR but PIL as RGB. Preserve the type
        # so a crop and the same PNG path use identical channel ordering.
        source = image.convert('RGB')
        width, height = image.size
    # RapidOCR mutates detector preprocessing state; the UI has multiple entry
    # points, so serialize lazy initialization and inference across sessions.
    with _INFERENCE_LOCK:
        raw, elapse = engine()(source)
    lines = []
    for row in (raw or []):
        quad, text, score = row[0], str(row[1]), float(row[2])
        if score < min_score or not text.strip():
            continue
        xs, ys = [float(point[0]) for point in quad], [float(point[1]) for point in quad]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        lines.append({'text':text.strip(), 'score':round(score, 4),
                      'quad':[[round(x, 1), round(y, 1)] for x, y in quad],
                      'bbox':[round(v, 1) for v in bbox],
                      'center':[round(sum(xs)/4, 1), round(sum(ys)/4, 1)],
                      'height':round(bbox[3]-bbox[1], 1)})
    lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
    result = {'lines':lines, 'elapsed_ms':round(sum(elapse)*1000, 1) if elapse else None}
    if width and height:
        result['image_size'] = [width, height]
    return result


def plain_text(result, joiner=' '):
    """把识别结果拼成纯文本，便于人工核对。"""
    return joiner.join(line['text'] for line in result.get('lines', []))
