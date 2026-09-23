"""诊断：打印文字遮罩后白色连通域的原始清单，判断箭头为何被剔除。

用法：
    .venv\\Scripts\\python.exe scripts\\debug_mask_components.py outputs\\ocr_samples\\08342_2.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from traffic_agent.arrows import mask_boxes, white_mask
from traffic_agent.ocr import read_text


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/ocr_samples/08342_2.jpg'
    with Image.open(path) as handle:
        image = handle.convert('RGB')
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    print(f'{path} size={width}x{height}')
    text = read_text(image)
    print('OCR bbox:')
    for line in text['lines']:
        print(f"   {line['text']!r} bbox={line['bbox']}")
    mask = white_mask(bgr)
    cleaned = mask_boxes(mask, [line['bbox'] for line in text['lines']], pad=0.06)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    rows = []
    for index in range(1, count):
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        w, h, area = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT], \
            int(stats[index, cv2.CC_STAT_AREA])
        rows.append((area, w, h, x, y))
    rows.sort(reverse=True)
    print(f'min_area={max(240, int(height*width*0.0012))}  连通域 {len(rows)} 个，按面积前 15：')
    for area, w, h, x, y in rows[:15]:
        component = (labels[y:y+h, x:x+w] == 0)
        print(f'   area={area:<7} bbox=[{x},{y},{x+w},{y+h}] w={w:<4} h={h:<4} fill={area/(w*h):.2f}')
    cv2.imwrite('outputs/arrow_probe/mask_debug.png', cleaned)


if __name__ == '__main__':
    main()
