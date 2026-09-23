"""统计数据集中各类别标注框的相对尺寸，用于验证 point-l / point-s 的“大/小”假设。

只读取 D:\\images-night 下 labels 目录里的 txt 与对应图片尺寸，不修改科研工程。
用法：
    .venv\\Scripts\\python.exe scripts\\inspect_label_sizes.py [labels_dir] [--limit N]
"""

import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

DEFAULT = Path(r'D:\images-night\from hainan\imagesnight')
NAMES = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s', 4:'aux'}


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    middle = len(values) // 2
    return {'n':len(values), 'median':round(values[middle], 4),
            'mean':round(sum(values) / len(values), 4),
            'p10':round(values[int(len(values) * 0.1)], 4),
            'p90':round(values[int(len(values) * 0.9)], 4)}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else DEFAULT
    limit = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--limit=')), 4000)
    areas, widths, heights = defaultdict(list), defaultdict(list), defaultdict(list)
    files = sorted((root/'train/labels').glob('*.txt')) + sorted((root/'val/labels').glob('*.txt'))
    used = 0
    for label in files[:limit * 4]:
        image = root/'train/images'/f'{label.stem}.jpg'
        if not image.is_file():
            image = root/'val/images'/f'{label.stem}.jpg'
        if not image.is_file():
            continue
        try:
            with Image.open(image) as handle:
                width, height = handle.size
        except Exception:
            continue
        used += 1
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            name = NAMES.get(int(parts[0]))
            if not name:
                continue
            bw, bh = float(parts[3]) * width, float(parts[4]) * height
            areas[name].append(bw * bh)
            widths[name].append(bw)
            heights[name].append(bh)
        if used >= limit:
            break
    report = {'labels_root':str(root), 'images_used':used,
              'classes':{name:{'box_area_px':stats(areas[name]), 'box_width_px':stats(widths[name]),
                               'box_height_px':stats(heights[name])} for name in NAMES.values() if areas[name]}}
    print(f'扫描图片数：{used}')
    header = f"{'class':8}{'n':>7}{'area median':>14}{'width median':>14}{'height median':>15}"
    print(header)
    for name in NAMES.values():
        if name not in report['classes']:
            continue
        row = report['classes'][name]
        print(f"{name:8}{row['box_area_px']['n']:>7}{row['box_area_px']['median']:>14.0f}"
              f"{row['box_width_px']['median']:>14.0f}{row['box_height_px']['median']:>15.0f}")
    out = Path(__file__).resolve().parents[1]/'outputs/label_size_report.json'
    import json
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'报告：{out}')


if __name__ == '__main__':
    main()
