"""在 OCR 文字遮罩之后做箭头检测，对照已知样本核对方向。

用法：
    .venv\\Scripts\\python.exe scripts\\probe_arrows2.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from traffic_agent.arrows import detect_arrows_white
from traffic_agent.config import OUTPUTS
from traffic_agent.ocr import read_text

# 人工看图确认的参考方向（用于核对方法是否可用，不构成完整评测集）
EXPECTED = {
    '08342_1.jpg': {'left', 'up', 'right'},
    '08342_2.jpg': {'left'},
    '08118_2.jpg': {'slight_left'},
    '08046_1.jpg': None,
    '08480_1.jpg': None,
}


def overlay(path, image, arrows, text_boxes):
    canvas = image.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(15, canvas.width//45))
    for box in text_boxes:
        draw.rectangle(box, outline='#38bdf8', width=2)
    for arrow in arrows:
        draw.rectangle(arrow['bbox'], outline='#f97316', width=max(2, canvas.width//320))
        draw.text((arrow['bbox'][0], max(0, arrow['bbox'][1]-font.size-3)),
                  f"{arrow['direction']} {arrow['confidence']}", fill='#f97316', font=font)
    canvas.save(path)


def main():
    manifest = json.loads((OUTPUTS/'ocr_samples.json').read_text(encoding='utf-8'))['crops']
    out_dir = OUTPUTS/'arrow_probe'
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in manifest:
        name = Path(item['crop']).name
        if name not in EXPECTED:
            continue
        with Image.open(item['crop']) as handle:
            image = handle.convert('RGB')
        text = read_text(image)
        text_boxes = [line['bbox'] for line in text['lines']]
        result = detect_arrows_white(image, text_boxes)
        got = [arrow['direction'] for arrow in result['arrows']]
        expected = EXPECTED[name]
        status = 'n/a' if expected is None else ('OK' if set(got) == expected else 'MISMATCH')
        overlay(out_dir/f'{Path(item["crop"]).stem}_arrows2.jpg', image, result['arrows'], text_boxes)
        print(f"--- {name} 期望={sorted(expected) if expected else '未标注'} 得到={got} -> {status}")
        print(f"    文字框 {len(text_boxes)}，箭头 {len(result['arrows'])}，"
              f"剔除统计 {result['rejected']}")
        records.append({'crop':item['crop'], 'expected':sorted(expected) if expected else None,
                        'got':got, 'status':status, 'arrows':result['arrows'],
                        'rejected':result['rejected'], 'text_lines':len(text_boxes)})
    path = OUTPUTS/'arrow_probe2.json'
    path.write_text(json.dumps({'records':records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n可视化：{out_dir}\n报告：{path}')


if __name__ == '__main__':
    main()
