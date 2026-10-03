"""Stage a small, leak-audited four-class fine-tuning experiment.

Existing source labels are read only. Private images, labels and provenance stay
under ignored outputs/. Numeric hundred-blocks and perceptual hashes are only
proxies for entity identity; this is not an independently reviewed test set.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from build_direction_dataset import night_exclusion_reason, phash

ROOT = Path(__file__).resolve().parents[1]
DAY = Path(r'D:\Private-dataset-master')
NIGHT = Path(r'D:\images-night\from hainan\imagesnight')
NAMES = ['wran', 'ban', 'point-l', 'point-s']


def read_label(path):
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        parsed = []
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                return None
            values = [float(value) for value in fields]
            class_id = int(values[0])
            if (values[0] != class_id or class_id not in range(5)
                    or not all(0 <= value <= 1 for value in values[1:3])
                    or not all(0 < value <= 1 for value in values[3:])):
                return None
            parsed.append((class_id, line))
        if not parsed or any(class_id == 4 for class_id, _ in parsed):
            return None
        return parsed
    except (OSError, ValueError):
        return None


def day_candidates(split, gold_images):
    label_index = defaultdict(list)
    for label in (DAY / 'labels').glob('*/*.txt'):
        label_index[label.stem].append(label)
    rows = []
    for image in (DAY / split / 'images').glob('*.jpg'):
        if str(image.resolve()).lower() in gold_images:
            continue
        labels = label_index.get(image.stem, [])
        if len(labels) != 1:
            continue
        parsed = read_label(labels[0])
        if parsed is None:
            continue
        block = (str(int(image.stem) // 100) if image.stem.isdigit()
                 else image.stem[:3])
        rows.append({'source': 'day', 'image': image, 'label': labels[0],
                     'group': block, 'classes': [item[0] for item in parsed]})
    return rows


def night_candidates(split, gold_images, allowed_groups):
    rows = []
    with (NIGHT / 'image_level_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        for metadata in csv.DictReader(handle):
            if metadata['split'] != split or metadata['group_id'] not in allowed_groups:
                continue
            image, label = Path(metadata['image']), Path(metadata['label'])
            if (str(image.resolve()).lower() in gold_images or
                    night_exclusion_reason(image, metadata) or
                    not image.is_file() or not label.is_file()):
                continue
            parsed = read_label(label)
            if parsed is None:
                continue
            rows.append({'source': 'night', 'image': image, 'label': label,
                         'group': metadata['group_id'],
                         'classes': [item[0] for item in parsed]})
    return rows


def rank(row, seed):
    return hashlib.sha256(f"{seed}:{row['image']}".encode()).digest()


def choose(rows, count, seed, used=None, cap=None):
    used = used if used is not None else set()
    taken = []
    groups = Counter()
    for row in sorted(rows, key=lambda item: rank(item, seed)):
        image = str(row['image'])
        if image in used or (cap is not None and groups[row['group']] >= cap):
            continue
        taken.append(row)
        groups[row['group']] += 1
        used.add(image)
        if len(taken) >= count:
            break
    return taken


def visual_identity(image):
    with Image.open(image) as opened:
        # Some camera exports use 0 for an unspecified orientation. Pixels and
        # YOLO coordinates remain in the stored frame for 0 and 1.
        if opened.getexif().get(274, 1) not in (0, 1):
            return None
        ratio = opened.width / opened.height
        visual_hash = phash(opened)
    return ratio, visual_hash


def too_close(identity, held_out):
    ratio, visual_hash = identity
    return any(abs(ratio / other_ratio - 1) < .15 and
               (visual_hash ^ other_hash).bit_count() <= 4
               for other_ratio, other_hash in held_out)


def stage(rows, split, out):
    manifest = []
    for row in rows:
        source = row['image']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        stem = f"{row['source']}_{source.stem}_{digest[:10]}"
        image_out = out / 'images' / split / f'{stem}{source.suffix.lower()}'
        label_out = out / 'labels' / split / f'{stem}.txt'
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_out)
        shutil.copy2(row['label'], label_out)
        manifest.append({'source': row['source'], 'source_image': str(source),
                         'source_label': str(row['label']), 'group': row['group'],
                         'sha256': digest, 'classes': row['classes'],
                         'staged_image': str(image_out), 'split': split})
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=Path, default=ROOT/'outputs/direction_upgrade_v3/combined35.gold.json')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/finetune_direction_20261004')
    parser.add_argument('--seed', type=int, default=20261004)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists; use a new --out for a new immutable split')
    gold = json.loads(args.gold.read_text(encoding='utf-8'))['cases']
    gold_images = {str(Path(case['source_image']).resolve()).lower() for case in gold}
    gold_day_blocks = {str(int(Path(case['source_image']).stem) // 100)
                       for case in gold if 'Private-dataset-master' in case['source_image']
                       and Path(case['source_image']).stem.isdigit()}
    manifest_rows = list(csv.DictReader((NIGHT/'image_level_manifest.csv').open(encoding='utf-8-sig')))
    groups = defaultdict(set)
    gold_night_groups = set()
    for row in manifest_rows:
        groups[row['group_id']].add(row['split'])
        if str(Path(row['image']).resolve()).lower() in gold_images:
            gold_night_groups.add(row['group_id'])
    train_groups = {name for name, splits in groups.items()
                    if name and splits == {'train'} and name not in gold_night_groups}
    val_groups = {name for name, splits in groups.items() if name and splits == {'val'}
                  and name not in gold_night_groups}
    day_val = day_candidates('val', gold_images)
    night_val = night_candidates('val', gold_images, val_groups)
    used = set()
    validation = (choose([row for row in day_val if 2 in row['classes']], 30, args.seed, used, cap=3)
                  + choose([row for row in day_val if 2 not in row['classes']], 20, args.seed+1, used, cap=3)
                  + choose([row for row in night_val if 2 in row['classes']], 15, args.seed+2, used, cap=8)
                  + choose([row for row in night_val if 2 not in row['classes']], 15, args.seed+3, used, cap=8))
    held_out = []
    for image in [Path(case['source_image']) for case in gold] + [row['image'] for row in validation]:
        identity = visual_identity(image)
        if identity is not None:
            held_out.append(identity)
    validation = [row for row in validation if visual_identity(row['image']) is not None]
    val_day_blocks = {row['group'] for row in validation if row['source'] == 'day'}
    forbidden_day_blocks = gold_day_blocks | val_day_blocks
    day_train = [row for row in day_candidates('train', gold_images)
                 if row['group'] not in forbidden_day_blocks]
    night_train = night_candidates('train', gold_images, train_groups)
    candidates = (choose([row for row in day_train if row['classes'].count(2) >= 2], 180,
                         args.seed+4, cap=8)
                  + choose([row for row in day_train if row['classes'].count(2) == 1], 160,
                           args.seed+5, cap=8)
                  + choose([row for row in day_train if 2 not in row['classes']], 100,
                           args.seed+6, cap=8)
                  + choose([row for row in night_train if 2 in row['classes']], 80,
                           args.seed+7, cap=8)
                  + choose([row for row in night_train if 2 not in row['classes']], 40,
                           args.seed+8, cap=8))
    training, seen_sha = [], set()
    rejected = Counter()
    for row in candidates:
        digest = hashlib.sha256(row['image'].read_bytes()).hexdigest()
        if digest in seen_sha:
            rejected['duplicate_train_sha'] += 1
            continue
        identity = visual_identity(row['image'])
        if identity is None or too_close(identity, held_out):
            rejected['orientation_or_near_holdout'] += 1
            continue
        seen_sha.add(digest)
        training.append(row)
    if not training or not validation:
        raise ValueError('No usable train or validation images after leakage audit')
    args.out.mkdir(parents=True)
    staged = stage(training, 'train', args.out) + stage(validation, 'val', args.out)
    yaml = (f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
            "names:\n" + ''.join(f"  {index}: {name}\n" for index,name in enumerate(NAMES)))
    (args.out/'data.yaml').write_text(yaml, encoding='utf-8')
    (args.out/'manifest.json').write_text(json.dumps(staged, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'split_status': 'development_proxy_groups_not_independently_verified',
               'gold_images_excluded_from_train': len(gold_images),
               'gold_day_hundred_blocks_excluded': len(gold_day_blocks),
               'night_train_only_groups': len(train_groups),
               'night_val_only_groups': len(val_groups),
               'train_images': len(training), 'val_images': len(validation),
               'train_by_source': dict(Counter(row['source'] for row in training)),
               'val_by_source': dict(Counter(row['source'] for row in validation)),
               'train_class_images': {str(i): sum(i in row['classes'] for row in training) for i in range(4)},
               'val_class_images': {str(i): sum(i in row['classes'] for row in validation) for i in range(4)},
               'rejected': dict(rejected)}
    (args.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
