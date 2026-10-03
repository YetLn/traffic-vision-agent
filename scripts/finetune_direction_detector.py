"""Run a bounded four-class fine-tuning experiment on a staged, audited split.

This script does not change the application's checkpoint. Compare the saved
candidate against both the coarse-label holdout and original-image direction
gold before considering deployment.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/traffic_best.pt')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--name', default='fine_tune')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 320:
        parser.error('epochs and batch must be positive; imgsz must be at least 320')
    summary = json.loads((args.split/'summary.json').read_text(encoding='utf-8'))
    if summary['split_status'] != 'development_proxy_groups_not_independently_verified':
        parser.error('Expected the audited development-proxy split')
    run_dir = args.split/'runs'/args.name
    if run_dir.exists():
        parser.error('Run already exists; use a new --name')
    settings = dict(data=str((args.split/'data.yaml').resolve()),
                    epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=0, cache=False,
                    optimizer='AdamW', lr0=0.0002, lrf=0.2,
                    warmup_epochs=0.5, seed=20261004, deterministic=True,
                    amp=False, mosaic=0.0, mixup=0.0,
                    degrees=3.0, translate=0.05, scale=0.20,
                    project=str((args.split/'runs').resolve()), name=args.name,
                    exist_ok=False, plots=False)
    print(json.dumps({'weights': str(args.weights.resolve()), 'settings': settings},
                     ensure_ascii=False, indent=2), flush=True)
    model = YOLO(str(args.weights.resolve()), task='detect')
    model.train(**settings)


if __name__ == '__main__':
    main()
