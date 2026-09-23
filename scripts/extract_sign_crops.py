"""从私有数据集筛选“大型指路标志（class 2）”样本，并按原图分辨率裁出牌面。

筛选规则（可通过参数调整）：
- 该 txt 中 class 2 记录数 >= --min-count；
- class 2 占该文件全部记录的比例 >= --min-share（默认 0.5，保证画面主体是指路牌）；
- class 2 记录中最大框的面积 >= --min-area 像素（默认 0，即按面积排序取前 N）。

裁剪始终使用**原图分辨率**，不使用缩放到 640 的推理图。

用法：
    .venv\\Scripts\\python.exe scripts\\extract_sign_crops.py --limit 120
    .venv\\Scripts\\python.exe scripts\\extract_sign_crops.py --min-count 2 --min-share 0.8 --limit 40
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(r'D:\Private-dataset-master')
POINT_L = 2
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
# 说明：images/<batch>/ 在本数据集里是空目录，原图实际位于 train/images、val/images、test/images。
SEARCH_DIRS = ('images', 'train/images', 'val/images', 'test/images')
OUT_DIR = Path(__file__).resolve().parents[1]/'outputs'


def label_files():
    for batch in sorted((ROOT/'labels').iterdir()):
        if batch.is_dir():
            for txt in sorted(batch.glob('*.txt')):
                yield txt


def find_image(stem):
    for folder in SEARCH_DIRS:
        base = ROOT/folder
        if not base.is_dir():
            continue
        for suffix in IMAGE_SUFFIXES:
            direct = base/f'{stem}{suffix}'
            if direct.is_file():
                return direct
            for batch in sorted(base.iterdir()):
                candidate = batch/f'{stem}{suffix}' if batch.is_dir() else None
                if candidate and candidate.is_file():
                    return candidate
    return None


def build_image_index():
    """一次遍历建立 id -> 原图路径 的索引，避免对每个标注文件重复扫描目录。"""
    index = {}
    for folder in SEARCH_DIRS:
        base = ROOT/folder
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(path.stem, path)
    return index


def read_boxes(txt):
    rows = []
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append({'cls':int(float(parts[0])), 'cx':float(parts[1]), 'cy':float(parts[2]),
                     'w':float(parts[3]), 'h':float(parts[4])})
    return rows


def main():
    parser = argparse.ArgumentParser(description='筛选并裁出大型指路标志样本')
    parser.add_argument('--limit', type=int, default=120, help='最多裁出多少张牌面')
    parser.add_argument('--min-count', type=int, default=1, help='class 2 记录数下限')
    parser.add_argument('--min-share', type=float, default=0.5, help='class 2 占比下限')
    parser.add_argument('--min-area', type=float, default=0, help='class 2 最大框面积下限（像素）')
    parser.add_argument('--padding', type=float, default=0.04, help='裁剪外扩比例')
    parser.add_argument('--device', default=None, help='保留参数位；裁剪不使用推理')
    args = parser.parse_args()

    crop_dir = OUT_DIR/'ocr_samples'
    crop_dir.mkdir(parents=True, exist_ok=True)
    index = build_image_index()
    print(f'原图索引：{len(index)} 张')
    candidates, missing_image, scanned = [], 0, 0
    for txt in label_files():
        scanned += 1
        rows = read_boxes(txt)
        point_l = [row for row in rows if row['cls'] == POINT_L]
        if len(point_l) < args.min_count or not point_l:
            continue
        share = len(point_l)/max(1, len(rows))
        if share < args.min_share:
            continue
        image_path = index.get(txt.stem)
        if image_path is None:
            missing_image += 1
            continue
        with Image.open(image_path) as handle:
            width, height = ImageOps.exif_transpose(handle).size
        boxes = []
        for row in point_l:
            bw, bh = row['w']*width, row['h']*height
            area = bw*bh
            if area < args.min_area:
                continue
            boxes.append({'cx':row['cx'], 'cy':row['cy'], 'w':row['w'], 'h':row['h'],
                          'area':round(area), 'width':round(bw), 'height':round(bh)})
        if not boxes:
            continue
        boxes.sort(key=lambda b: -b['area'])
        candidates.append({'id':txt.stem, 'batch':txt.parent.name, 'label_file':str(txt),
                           'image':str(image_path),
                           'image_size':[width, height], 'class2_share':round(share, 3),
                           'records':len(rows), 'class2_boxes':boxes, 'max_area':boxes[0]['area']})

    candidates.sort(key=lambda c: -c['max_area'])
    selected = candidates[:args.limit]
    saved = []
    for item in selected:
        with Image.open(item['image']) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
            width, height = image.size
            # 文件名必须带批次号：本数据集不同批次目录存在**重复 id 且是不同图片**，
            # 只用 id 命名会互相覆盖（实测导致 6 个案例中 4 个取到另一张同 id 牌面）。
            batch = str(item['batch']).replace('/', '_')
            for index, box in enumerate(item['class2_boxes'][:3], start=1):
                pad_x, pad_y = box['w']*width*args.padding, box['h']*height*args.padding
                left = max(0, int((box['cx']-box['w']/2)*width - pad_x))
                top = max(0, int((box['cy']-box['h']/2)*height - pad_y))
                right = min(width, int((box['cx']+box['w']/2)*width + pad_x))
                bottom = min(height, int((box['cy']+box['h']/2)*height + pad_y))
                if right-left < 16 or bottom-top < 16:
                    continue
                name = f"{batch}_{item['id']}_{index}.jpg"
                crop = image.crop((left, top, right, bottom))
                crop.save(crop_dir/name, quality=95)
                saved.append({'id':item['id'], 'batch':item['batch'], 'crop':str(crop_dir/name),
                              'image':item['image'], 'bbox':[left, top, right, bottom],
                              'crop_size':list(crop.size),
                              'source_image_size':[width, height],
                              'bbox_share_of_image':round((right-left)*(bottom-top)/(width*height), 4)})

    report = {'scanned_label_files':scanned, 'candidates':len(candidates),
              'missing_image':missing_image, 'crops_saved':len(saved),
              'filter':{'min_count':args.min_count, 'min_share':args.min_share, 'min_area':args.min_area},
              'top_candidates':[{k:v for k, v in item.items() if k != 'class2_boxes'} |
                                {'class2_count':len(item['class2_boxes'])} for item in selected[:30]],
              'crops':saved}
    path = OUT_DIR/'ocr_samples.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"扫描标注文件 {scanned}，满足条件 {len(candidates)}，缺原图 {missing_image}，裁出牌面 {len(saved)}")
    print(f"裁剪目录：{crop_dir}")
    print(f"清单：{path}")
    for item in selected[:10]:
        print(f"  {item['id']}  原图{item['image_size']}  2类框{item['class2_boxes'][0]['width']}x"
              f"{item['class2_boxes'][0]['height']}  占比{item['class2_share']}")


if __name__ == '__main__':
    main()
