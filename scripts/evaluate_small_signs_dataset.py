"""Explore the small-sign prototype on raw validation images from both datasets.

Only existing coarse YOLO labels are used. The report can measure coarse-box
recall and coverage/rejection, but cannot measure fine-grained sign accuracy.
All per-image data stays under outputs/ and is excluded from Git.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.small_sign_reader import read_small_signs


DAY = Path(r'D:\Private-dataset-master')
NIGHT = Path(r'D:\images-night\from hainan\imagesnight')
CLASSES = (0, 1, 3)


def _stable_rank(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def _night_originals(night_root=NIGHT):
    manifest = night_root / 'image_level_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != 'val' or row.get('sample_type', '').strip():
                continue
            provenance = ' '.join(str(row.get(key, '')).lower() for key in
                                  ('image', 'source_image', 'original_stem'))
            if any(token in provenance for token in
                   ('light_attack', '_existing_source_attack', '_global_side_height_radius')):
                continue
            image = Path(row['image'])
            label = Path(row['label'])
            if image.is_file() and label.is_file():
                yield {'source': 'night', 'image': image, 'label': label,
                       'group': row.get('group_id') or image.stem}


def _day_originals(day_root=DAY):
    index = {}
    ambiguous = set()
    for label in (day_root / 'labels').glob('*/*.txt'):
        if label.stem in index:
            ambiguous.add(label.stem)
        index[label.stem] = label
    for image in sorted((day_root / 'val/images').glob('*.jpg')):
        label = index.get(image.stem)
        if label and image.stem not in ambiguous:
            yield {'source': 'day', 'image': image, 'label': label,
                   'group': image.stem}


def _read_boxes(label, size):
    width, height = size
    boxes = []
    for line in label.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            class_id = int(float(fields[0]))
            cx, cy, bw, bh = [float(value) for value in fields[1:5]]
        except ValueError:
            continue
        if class_id not in CLASSES or bw <= 0 or bh <= 0:
            continue
        box = [max(0, (cx-bw/2)*width), max(0, (cy-bh/2)*height),
               min(width, (cx+bw/2)*width), min(height, (cy+bh/2)*height)]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append({'class_id': class_id, 'bbox': box,
                          'short_side': min(box[2]-box[0], box[3]-box[1]),
                          'ratio': (box[2]-box[0])/(box[3]-box[1])})
    return boxes


def _size_bin(short_side):
    if short_side < 160:
        return '96-159'
    if short_side < 300:
        return '160-299'
    return '300+'


def inventory(day_root=DAY, night_root=NIGHT):
    records = []
    for row in list(_day_originals(day_root)) + list(_night_originals(night_root)):
        try:
            with Image.open(row['image']) as handle:
                size = handle.size
                if handle.getexif().get(274, 1) != 1:
                    continue
        except OSError:
            continue
        boxes = _read_boxes(row['label'], size)
        if not boxes:
            continue
        eligible = [box for box in boxes if box['short_side'] >= 96
                    and 0.72 <= box['ratio'] <= 1.38]
        records.append({**row, 'image_size': size, 'boxes': boxes,
                        'eligible': eligible})
    return records


def select(records, per_class=20, controls=10, seed=20260926):
    chosen = []
    used = set()
    for source in ('day', 'night'):
        source_rows = [r for r in records if r['source'] == source]
        for class_id in CLASSES:
            pool = [r for r in source_rows if any(b['class_id'] == class_id
                                                  for b in r['eligible'])]
            buckets = {'96-159': [], '160-299': [], '300+': []}
            for row in pool:
                biggest = max(b['short_side'] for b in row['eligible']
                              if b['class_id'] == class_id)
                buckets[_size_bin(biggest)].append(row)
            for bucket in buckets.values():
                bucket.sort(key=lambda r: _stable_rank(str(r['image']), seed))
            selected = []
            # Round-robin size buckets; prefer new capture groups on the first pass.
            groups = set()
            while len(selected) < per_class and any(buckets.values()):
                progress = False
                for key in ('96-159', '160-299', '300+'):
                    bucket = buckets[key]
                    if not bucket:
                        continue
                    index = next((i for i, row in enumerate(bucket)
                                  if row['image'] not in used and row['group'] not in groups), None)
                    if index is None:
                        index = next((i for i, row in enumerate(bucket)
                                      if row['image'] not in used), None)
                    if index is None:
                        bucket.clear()
                        continue
                    row = bucket.pop(index)
                    selected.append(row)
                    used.add(row['image'])
                    groups.add(row['group'])
                    progress = True
                    if len(selected) >= per_class:
                        break
                if not progress:
                    break
            for row in selected:
                chosen.append({**row, 'stratum': f'{source}_{class_id}'})
        control_pool = [r for r in source_rows if not r['eligible']
                        and r['image'] not in used]
        control_pool.sort(key=lambda r: _stable_rank(str(r['image']), seed))
        for row in control_pool[:controls]:
            chosen.append({**row, 'stratum': f'{source}_small_control'})
            used.add(row['image'])
    return chosen


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right-left)*max(0, bottom-top)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter/max(area_a+area_b-inter, 1e-9)


def evaluate(selected, detector, confidence=0.25):
    results = []
    for index, row in enumerate(selected, 1):
        with Image.open(row['image']) as handle:
            report = read_small_signs(handle, detector, confidence)
        gt = []
        for box in row['eligible']:
            overlap = max((_iou(box['bbox'], sign['bbox']) for sign in report['signs']
                           if sign['class_id'] == box['class_id']), default=0.0)
            gt.append({'class_id': box['class_id'], 'size_bin': _size_bin(box['short_side']),
                       'matched_iou50': overlap >= 0.5, 'best_iou': round(overlap, 3)})
        results.append({'source': row['source'], 'stratum': row['stratum'],
                        'image': str(row['image']), 'group': row['group'],
                        'eligible_gt': gt, 'all_coarse_boxes': len(row['boxes']),
                        'detected': len(report['signs']),
                        'accepted': report['accepted'], 'abstained': report['abstained'],
                        'signs': report['signs']})
        if index % 10 == 0 or index == len(selected):
            print(f'processed {index}/{len(selected)}', flush=True)
    return results


def summarize(records, selected, results):
    summary = {'inventory_images': dict(Counter(r['source'] for r in records)),
               'sampled_images': dict(Counter(r['source'] for r in selected)),
               'sampled_groups': {source: len({r['group'] for r in selected
                                               if r['source'] == source})
                                  for source in ('day', 'night')},
               'strata': dict(Counter(r['stratum'] for r in selected)),
               'eligible_gt': {}, 'detected_boxes': {}, 'accepted_signs': {},
               'rejection_reasons': {}, 'accepted_names': {}}
    for source in ('day', 'night'):
        rows = [r for r in results if r['source'] == source]
        gt = [b for r in rows for b in r['eligible_gt']]
        signs = [s for r in rows for s in r['signs']]
        summary['eligible_gt'][source] = {
            'total': len(gt), 'matched_iou50': sum(b['matched_iou50'] for b in gt),
            'by_class': {str(c): {'total': sum(b['class_id'] == c for b in gt),
                                  'matched_iou50': sum(b['class_id'] == c and b['matched_iou50']
                                                       for b in gt)} for c in CLASSES}}
        summary['detected_boxes'][source] = len(signs)
        summary['accepted_signs'][source] = sum(s['accepted'] for s in signs)
        summary['rejection_reasons'][source] = dict(Counter(s.get('reason', '') for s in signs
                                                            if not s['accepted']))
        summary['accepted_names'][source] = dict(Counter(s['name'] for s in signs
                                                         if s['accepted']))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=20)
    parser.add_argument('--controls', type=int, default=10)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--day-root', type=Path, default=DAY)
    parser.add_argument('--night-root', type=Path, default=NIGHT)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out', type=Path,
                        default=ROOT/'outputs/small_sign_eval_20260926.json')
    parser.add_argument('--summary-out', type=Path,
                        default=ROOT/'reports/small-sign-eval-20260926-summary.json')
    args = parser.parse_args()
    records = inventory(args.day_root, args.night_root)
    selected = select(records, args.per_class, args.controls, args.seed)
    print('selected', len(selected), dict(Counter(r['stratum'] for r in selected)), flush=True)
    if args.dry_run:
        return
    detector = TrafficSignDetector()
    results = evaluate(selected, detector, args.confidence)
    summary = summarize(records, selected, results)
    payload = {'selection': {'per_class': args.per_class, 'controls': args.controls,
                             'seed': args.seed, 'confidence': args.confidence,
                             'source_splits': {'day': 'val', 'night': 'val originals only'}},
               'summary': summary, 'samples': results,
               'warning': 'No fine-grained ground truth; accepted subtype counts are coverage, not accuracy.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps({'selection': payload['selection'],
                                            'summary': summary,
                                            'warning': payload['warning']},
                                           ensure_ascii=False, indent=2)+'\n',
                                encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('local_report', args.out, flush=True)
    print('aggregate_summary', args.summary_out, flush=True)


if __name__ == '__main__':
    main()
