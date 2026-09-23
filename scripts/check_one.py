"""逐个导入，定位原生库崩溃点。"""

import importlib
import sys

name = sys.argv[1]
module = importlib.import_module(name)
print('ok', name, getattr(module, '__version__', '?'), flush=True)
