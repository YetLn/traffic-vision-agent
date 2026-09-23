"""量测牌面像素分布：文字区与背景区的灰度、饱和度对比。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/ocr_samples/08342_2.jpg'
with Image.open(path) as handle:
    image = handle.convert('RGB')
bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
print(f'{path} size={image.size}')
print('整图灰度: min', gray.min(), 'max', gray.max(), 'mean', round(float(gray.mean()), 1),
      'median', int(np.median(gray)))
print('整图 V 通道: mean', round(float(hsv[:, :, 2].mean()), 1),
      'S 通道: mean', round(float(hsv[:, :, 1].mean()), 1))
otsu_value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Otsu 阈值', round(otsu_value, 1), '前景占比', round(float((otsu > 0).mean()), 3))

# 取文字框与箭头区域的像素统计
for name, box in [('text 平安路', (362, 56, 747, 172)), ('text Pingan Rd', (382, 176, 747, 246)),
                  ('arrow(左箭头区域)', (60, 30, 340, 200)),
                  ('sign-blue 背景', (400, 100, 450, 140))]:
    x0, y0, x1, y1 = box
    region_gray = gray[y0:y1, x0:x1]
    region_hsv = hsv[y0:y1, x0:x1]
    bright = region_gray > np.percentile(region_gray, 90)
    print(f'{name}: 灰度 p10={int(np.percentile(region_gray,10))} p50={int(np.percentile(region_gray,50))} '
          f'p90={int(np.percentile(region_gray,90))} max={region_gray.max()} '
          f'| 最亮10%像素 V中位={int(np.median(region_hsv[:,:,2][bright]))} '
          f'S中位={int(np.median(region_hsv[:,:,1][bright]))}')
cv2.imwrite('outputs/arrow_probe/gray_debug.png', gray)
