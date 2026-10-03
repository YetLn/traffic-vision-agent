"""Compare coarse YOLO box recall on the same local original-road images.

This is a development diagnostic: direction gold annotates selected boards,
and coarse labels may be incomplete. Unmatched detections are not false positives.
No annotation box enters inference and no source image path enters the summary.
"""

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_small_signs_dataset import _iou, inventory, select
from traffic_agent.config import WEIGHTS
from traffic_agent.detector import TrafficSignDetector


def targets(direction_gold, seed, per_class, controls):
    selected = select(inventory(), per_class=per_class, controls=controls, seed=seed)
    records = defaultdict(list)
    for row in selected:
        for box in row['eligible']:
            records[str(row['image'])].append((f"small_{row['source']}_{box['class_id']}",
                                               box['class_id'], box['bbox']))
    cases = json.loads(direction_gold.read_text(encoding='utf-8'))['cases']
    for case in cases:
        for sign in case['signs']:
            records[str(case['source_image'])].append(('direction', 2, sign['bbox']))
    return records, len(selected), len(cases)


def measure(records, weight_path):
    detector = TrafficSignDetector(weight_path=weight_path)
    counts = defaultdict(lambda: {'gt': 0, 'hit': 0})
    detected = defaultdict(int)
    elapsed = []
    for index, (image_path, boxes) in enumerate(records.items(), 1):
        with Image.open(image_path) as image:
            result = detector.detect(image, conf=0.25)
        elapsed.append(result['elapsed_ms'])
        for prediction in result['detections']:
            detected[str(prediction['class_id'])] += 1
        for group, class_id, box in boxes:
            overlap = max((_iou(box, row['bbox']) for row in result['detections']
                           if row['class_id'] == class_id), default=0.0)
            counts[group]['gt'] += 1
            counts[group]['hit'] += overlap >= 0.5
        if index % 30 == 0 or index == len(records):
            print(weight_path.name, index, '/', len(records), flush=True)
    del detector
    gc.collect()
    return {'weights_sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest(),
            'counts': dict(counts), 'detected_boxes_by_class': dict(detected),
            'mean_inference_ms': round(sum(elapsed) / len(elapsed), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction-gold', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                        help='Additional checkpoint; repeat for each candidate')
    parser.add_argument('--per-class', type=int, default=20)
    parser.add_argument('--controls', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    models = {'current_v12': WEIGHTS}
    for item in args.checkpoint:
        name, separator, path = item.partition('=')
        if not separator or not name or name in models or not Path(path).is_file():
            parser.error('--checkpoint must be a unique NAME=existing-path')
        models[name] = Path(path)
    records, small_images, direction_images = targets(args.direction_gold, args.seed,
                                                       args.per_class, args.controls)
    if not records:
        raise ValueError('No eligible labeled images found')
    report = {'scope': 'development_coarse_box_recall_only',
              'selection': {'small_source_images': small_images,
                            'direction_source_images': direction_images,
                            'unique_source_images': len(records), 'seed': args.seed,
                            'per_class': args.per_class, 'controls': args.controls},
              'metric': 'Each eligible gold box has a same-class predicted box with IoU >= 0.5; '
                        'not one-to-one. Fine-grained accuracy and false-positive rate are unknown.',
              'timing_note': 'Mean inference times are run-specific and not used for checkpoint selection.',
              'checkpoints': {}}
    for name, path in models.items():
        print('model', name, flush=True)
        report['checkpoints'][name] = measure(records, path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print('summary', args.out, flush=True)


if __name__ == '__main__':
    main()
