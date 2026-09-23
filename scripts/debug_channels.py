"""量测文字/箭头与蓝底背景的通道差异，用于确定前景提取判据。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/ocr_samples/08342_2.jpg'
with Image.open(path) as handle:
    image = handle.convert('RGB')
bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

REGIONS = {
    'text 平安路': (360, 50, 750, 180),
    'text Pingan Rd': (380, 175, 750, 250),
    'text (迎宾路)': (400, 250, 745, 345),
    'arrow left': (110, 120, 350, 230),
    'blue background': (400, 100, 460, 150),
}
print(f'{path}\n{"region":18}{"gray p50":>10}{"gray p90":>10}{"sat p50":>9}{"sat p90":>9}'
      f'{"B p50":>8}{"G p50":>8}{"R p50":>8}{"B-G":>7}')
for name, (x0, y0, x1, y1) in REGIONS.items():
    g = gray[y0:y1, x0:x1]
    ss = s[y0:y1, x0:x1]
    region = bgr[y0:y1, x0:x1]
    b50, g50, r50 = (int(np.median(region[:, :, i])) for i in range(3))
    print(f'{name:18}{int(np.median(g)):>10}{int(np.percentile(g,90)):>10}'
          f'{int(np.median(ss)):>9}{int(np.percentile(ss,90)):>9}'
          f'{b50:>8}{g50:>8}{r50:>8}{b50-g50:>7}')

# 判据候选：低饱和（灰/白）与高亮度
for name, mask in [
    ('sat<120', (s < 120).astype(np.uint8)*255),
    ('sat<120 & V>110', ((s < 120) & (v > 110)).astype(np.uint8)*255),
    ('B-G<60', ((bgr[:, :, 0].astype(int)-bgr[:, :, 1].astype(int)) < 60).astype(np.uint8)*255),
    ('V>140', (v > 140).astype(np.uint8)*255),
]:
    ys, xs = np.nonzero(mask)
    print(f'{name:18} 占比={mask.mean()/255:.3f} 前景灰度均值='
          f'{gray[mask > 0].mean():.1f}' if mask.any() else f'{name}: 空')
