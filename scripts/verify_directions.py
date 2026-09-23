"""Development-only original-image run. Never substitutes annotation boxes."""
import argparse
import hashlib
import json
import sys
import re
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import WEIGHTS
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.direction_reader import read_directions, draw_evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root=Path(__file__).resolve().parents[1]
    modules=('traffic_agent/direction_reader.py','traffic_agent/direction_geometry.py',
             'traffic_agent/direction_groups.py','traffic_agent/panel_detector.py','traffic_agent/ocr.py',
             'traffic_agent/perspective.py','traffic_agent/ocr_refinement.py')
    implementation_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in modules}
    gold = json.loads(args.gold.read_text(encoding='utf-8'))
    cases = gold['cases']
    ids = [c['case_id'] for c in cases]
    if not ids or any(not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',value) for value in ids):
        raise ValueError('Gold case IDs must be safe nonempty file names')
    if len(ids) != len(set(ids)):
        raise ValueError('Gold case IDs must be nonempty and unique')
    for case in cases:
        with Image.open(case['source_image']) as image:
            image.verify()
    detector = TrafficSignDetector()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in cases:
        # Only source_image is read. No labels, expected relations or gold boxes
        # are passed into the inference path.
        with Image.open(case['source_image']) as image:
            report = read_directions(image, detector)
            draw_evidence(image, report, args.out/(case['case_id']+'.jpg'))
        report['source_sha256'] = hashlib.sha256(Path(case['source_image']).read_bytes()).hexdigest()
        results[case['case_id']] = report
        print(case['case_id'], 'signs', len(report['signs']), 'relations',
              sum(len(s['relations']) for s in report['signs']), flush=True)
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha for name,sha in implementation_hashes.items()):
        raise RuntimeError('Implementation changed during verification; rerun before publishing metrics')
    payload = {'run_type': 'development_original_image_yolo',
               'weights_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
               'implementation_sha256': implementation_hashes,
               'predictions': results}
    (args.out/'predictions.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
