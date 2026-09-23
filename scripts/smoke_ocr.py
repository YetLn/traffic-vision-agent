"""最小 OCR 冒烟测试：对指定图片文件直接跑 RapidOCR，打印原始返回结构。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

files = sys.argv[1:] or ['outputs/ocr_samples/08342_1.jpg']
print('rapidocr import...', flush=True)
from rapidocr_onnxruntime import RapidOCR
print('class ok', flush=True)
start = time.perf_counter()
engine = RapidOCR()
print(f'engine init {round((time.perf_counter()-start)*1000)} ms', flush=True)
for name in files:
    path = Path(name)
    print(f'=== {path} exists={path.is_file()}', flush=True)
    if not path.is_file():
        continue
    start = time.perf_counter()
    raw, elapse = engine(str(path))
    print(f'  elapsed {round((time.perf_counter()-start)*1000)} ms; elapse={elapse}', flush=True)
    print(f'  raw type={type(raw)} len={len(raw) if raw else 0}', flush=True)
    for row in (raw or [])[:40]:
        print('   ', row, flush=True)
