"""指路牌解析验证：跑「文字分组 + 箭头候选 + 位置规则」，输出可视化与结构化结果。

用法：
    .venv\\Scripts\\python.exe scripts\parse_signs.py --limit 10
    .venv\\Scripts\\python.exe scripts\parse_signs.py --names 08342_2.jpg,08342_1.jpg
结果写入 outputs/sign_parse.json，可视化写入 outputs/sign_parse/。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from traffic_agent.config import OUTPUTS
from traffic_agent.sign_parser import parse_sign


def draw(image, result, path):
    canvas = image.convert('RGB').copy()
    painter = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(15, canvas.width//45))
    for group in result['groups']:
        painter.rectangle(group['bbox'], outline='#38bdf8', width=2)
    for route in result['routes']:
        painter.rectangle(route['arrow_bbox'], outline='#22c55e', width=3)
        painter.text((route['arrow_bbox'][0], max(0, route['arrow_bbox'][1]-font.size-3)),
                     route['direction'], fill='#22c55e', font=font)
    for item in result['uncertain']:
        if 'arrow_bbox' in item:
            painter.rectangle(item['arrow_bbox'], outline='#ef4444', width=3)
            painter.text((item['arrow_bbox'][0], max(0, item['arrow_bbox'][1]-font.size-3)),
                         '?', fill='#ef4444', font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description='指路牌解析验证')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--names', default='')
    args = parser.parse_args()
    manifest = json.loads((OUTPUTS/'ocr_samples.json').read_text(encoding='utf-8'))['crops']
    if args.names:
        wanted = set(args.names.split(','))
        samples = [item for item in manifest if Path(item['crop']).name in wanted]
    else:
        samples = manifest[:args.limit]
    out_dir = OUTPUTS/'sign_parse'
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in samples:
        with Image.open(item['crop']) as handle:
            image = handle.convert('RGB')
        result = parse_sign(image)
        draw(image, result, out_dir/f"{Path(item['crop']).stem}_parsed.jpg")
        summary = '; '.join(f"{route['destination']}→{route['direction_label']}"
                            for route in result['routes']) or '无可用路线'
        print(f"{Path(item['crop']).name:>14} 文字组{len(result['groups'])} 箭头{result['arrow_count']} "
              f"路线{len(result['routes'])} 不确定{len(result['uncertain'])} | {summary[:70]}")
        records.append({'crop':item['crop'], **result})
    (OUTPUTS/'sign_parse.json').write_text(
        json.dumps({'samples':len(records), 'records':records}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'\n可视化：{out_dir}\n报告：{OUTPUTS/"sign_parse.json"}')


if __name__ == '__main__':
    main()
