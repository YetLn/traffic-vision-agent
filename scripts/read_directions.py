"""Run real YOLO -> OCR -> arrow evidence on a full road image, entirely locally."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import WEIGHTS, OUTPUTS
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.direction_reader import read_directions, draw_evidence, summarize_directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=OUTPUTS/'direction_demo')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    detector = TrafficSignDetector()
    with Image.open(args.image) as image:
        report = read_directions(image, detector, args.conf)
        args.out.mkdir(parents=True, exist_ok=True)
        draw_evidence(image, report, args.out/'evidence.jpg')
    report.update(source_image=str(args.image.resolve()),
                  source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest())
    (args.out/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summarize_directions(report))
    print(str(args.out.resolve()))


if __name__ == '__main__':
    main()
