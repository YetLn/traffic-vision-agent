"""箭头检测验证：在牌面裁剪图上跑箭头检测，并输出带标注的可视化图供人工核对。

用法：
    .venv\\Scripts\\python.exe scripts\\probe_arrows.py --ids 08342_1,08342_2,08118_2
    .venv\\Scripts\\python.exe scripts\\probe_arrows.py --limit 12
可视化结果写入 outputs/arrow_probe/。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from traffic_agent.arrows import detect_arrows
from traffic_agent.config import OUTPUTS


def overlay(image, arrows, path):
    canvas = image.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(16, canvas.width // 40))
    for index, arrow in enumerate(arrows, start=1):
        box = arrow['bbox']
        draw.rectangle(box, outline='#f97316', width=max(2, canvas.width // 300))
        label = f"{index}:{arrow['direction']} {arrow['confidence']}"
        draw.text((box[0], max(0, box[1]-font.size-4)), label, fill='#f97316', font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description='箭头检测验证')
    parser.add_argument('--ids', default='')
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()

    manifest = json.loads((OUTPUTS/'ocr_samples.json').read_text(encoding='utf-8'))['crops']
    out_dir = OUTPUTS/'arrow_probe'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.ids:
        names = {f'{name}.jpg' if not name.endswith('.jpg') else name for name in args.ids.split(',')}
        samples = [item for item in manifest if Path(item['crop']).name in names]
    else:
        samples = manifest[:args.limit]

    records = []
    for item in samples:
        with Image.open(item['crop']) as handle:
            image = handle.convert('RGB')
        result = detect_arrows(image)
        overlay(image, result['arrows'], out_dir/f"{Path(item['crop']).stem}_arrows.jpg")
        summary = ', '.join(f"{a['direction']}({a['confidence']})" for a in result['arrows'])
        print(f"{Path(item['crop']).name:>14} 白占比{result['white_ratio']:<7} "
              f"{len(result['arrows'])} 个候选箭头  {summary}")
        records.append({'crop':item['crop'], 'white_ratio':result['white_ratio'],
                        'arrows':result['arrows']})
    path = OUTPUTS/'arrow_probe.json'
    path.write_text(json.dumps({'samples':len(records), 'records':records}, ensure_ascii=False,
                               indent=2), encoding='utf-8')
    print(f'\n可视化：{out_dir}\n报告：{path}')


if __name__ == '__main__':
    main()
