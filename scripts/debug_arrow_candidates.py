"""列出箭头候选（未做最终过滤）的尺寸与形状，判断阈值是否把真箭头滤掉了。

用法：
    .venv\\Scripts\\python.exe scripts\\debug_arrow_candidates.py --names 08342_1.jpg,08118_2.jpg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from traffic_agent.arrows import _components, _direction, mask_boxes, white_mask
from traffic_agent.config import OUTPUTS
from traffic_agent.ocr import read_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', default='08342_1.jpg,08342_2.jpg,08118_2.jpg')
    parser.add_argument('--top', type=int, default=12)
    args = parser.parse_args()
    for name in args.names.split(','):
        path = OUTPUTS/'ocr_samples'/name
        if not path.is_file():
            print(f'{name}: 缺文件')
            continue
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        lines = read_text(image)['lines']
        mask = white_mask(bgr)
        cleaned = mask_boxes(mask, [line['bbox'] for line in lines], pad=0.06)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        rows = []
        for index in range(1, count):
            x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
            w, h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 150:
                continue
            component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else None
            hull = cv2.contourArea(cv2.convexHull(contour)) if contour is not None else 0
            solidity = cv2.contourArea(contour)/hull if hull > 0 else 0
            direction, confidence = _direction(component)
            rows.append((area, w, h, round(area/(w*h), 2), round(solidity, 2), direction,
                         round(confidence, 2), x, y))
        rows.sort(reverse=True)
        print(f'--- {name} {width}x{height}  min_area={max(240, int(height*width*0.0012))}')
        print(f'    {"area":>7}{"w":>5}{"h":>5}{"fill":>6}{"solid":>6}  dir/conf        pos')
        for area, w, h, fill, solidity, direction, confidence, x, y in rows[:args.top]:
            print(f'    {area:>7}{w:>5}{h:>5}{fill:>6}{solidity:>6}  {direction:<9}{confidence:<5} ({x},{y})')


if __name__ == '__main__':
    main()
