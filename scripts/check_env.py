"""依赖自检：OCR 安装后确认关键库仍可导入，避免版本冲突悄悄破坏原有链路。"""

import importlib
import sys

MODULES = ['onnxruntime', 'rapidocr_onnxruntime', 'cv2', 'numpy', 'PIL',
           'torch', 'ultralytics', 'gradio', 'yaml']

print('python', sys.version.split()[0])
failed = []
for name in MODULES:
    try:
        module = importlib.import_module(name)
        print(f'{name}: {getattr(module, "__version__", "?")}')
    except Exception as exc:
        failed.append(name)
        print(f'{name}: FAIL {type(exc).__name__}: {exc}')
print('FAILED:', failed or 'none')
