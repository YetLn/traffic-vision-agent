import argparse
import json
from pathlib import Path
from PIL import Image
from traffic_agent.config import ROOT, OUTPUTS
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.agent import ask


def main():
    parser = argparse.ArgumentParser(description='交通场景视觉查询原型')
    parser.add_argument('--image', type=Path, default=ROOT/'test_images/night_01.jpg')
    parser.add_argument('--question', default='图里有什么交通标志？帮我框出来，有多少个？')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    detector = TrafficSignDetector(device=args.device)
    with Image.open(args.image) as image:
        result = ask(detector, image, args.question, args.conf)
    report = {k:v for k,v in result.items() if k != 'state'}
    path = OUTPUTS/'result.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
