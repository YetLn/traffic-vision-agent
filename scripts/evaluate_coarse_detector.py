"""Compare class-2 box recall on the staged capture-group holdout.

The source annotations are coarse and may be incomplete, so reported false
positives are diagnostics rather than claims about semantic correctness.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(area_a+area_b-intersection, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in json.loads((args.split/'manifest.json').read_text(encoding='utf-8'))
            if row['split'] == 'val']
    model = YOLO(str(args.weights.resolve()), task='detect')
    counts = Counter()
    misses = []
    for index, row in enumerate(rows, 1):
        image_path = Path(row['staged_image'])
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        width, height = image.size
        label_path = args.split/'labels/val'/f'{image_path.stem}.txt'
        gt = []
        for line in label_path.read_text(encoding='utf-8-sig').splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            if int(cls) == 2:
                gt.append(([width*(cx-w/2), height*(cy-h/2),
                            width*(cx+w/2), height*(cy+h/2)],
                           (w*width)/(h*height) >= 2))
        prediction = model.predict(image, conf=0.25, imgsz=640,
                                   device='cpu', verbose=False)[0]
        predicted = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in prediction.boxes
                     if int(box.cls[0]) == 2]
        pairs = sorted(((iou(truth[0], box), gi, pi)
                        for gi, truth in enumerate(gt)
                        for pi, (box, _) in enumerate(predicted)), reverse=True)
        matched_gt, matched_pred = set(), set()
        for overlap, gi, pi in pairs:
            if overlap < 0.5:
                break
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
        counts['images'] += 1
        if gt:
            counts['images_with_class2'] += 1
            counts['images_with_class2_found'] += bool(matched_gt)
        counts['class2_gt'] += len(gt)
        counts['class2_matched'] += len(matched_gt)
        counts['class2_predictions'] += len(predicted)
        counts['class2_unmatched_predictions'] += len(predicted)-len(matched_pred)
        for gi, (_, wide) in enumerate(gt):
            key = 'wide' if wide else 'other'
            counts[f'{key}_gt'] += 1
            counts[f'{key}_matched'] += gi in matched_gt
            counts[f'{row["source"]}_{key}_gt'] += 1
            counts[f'{row["source"]}_{key}_matched'] += gi in matched_gt
            if gi not in matched_gt:
                misses.append({'image': row['source_image'], 'source': row['source'],
                               'wide': wide, 'best_iou': round(max(
                                   (iou(gt[gi][0], box) for box, _ in predicted),
                                   default=0), 3)})
        if index % 20 == 0:
            print(f'{index}/{len(rows)}', flush=True)
    metrics = {'counts': dict(counts),
               'recall': {key: (counts[f'{key}_matched']/counts[f'{key}_gt']
                                if counts[f'{key}_gt'] else None)
                          for key in ('class2', 'wide', 'other', 'day_wide',
                                      'night_wide', 'day_other', 'night_other')},
               'misses': misses,
               'notes': 'Development proxy holdout; IoU >= 0.5; original 640px inference; '
                        'wide means pixel box width/height >= 2; labels may be incomplete.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n',
                           encoding='utf-8')
    print(json.dumps({'counts': metrics['counts'], 'recall': metrics['recall']},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
