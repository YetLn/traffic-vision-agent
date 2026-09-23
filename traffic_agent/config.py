import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
RUNTIME.mkdir(exist_ok=True)
# Local credentials are excluded from version control; process env takes precedence.
local_config = RUNTIME / 'deepseek.json'
if local_config.is_file():
    settings = json.loads(local_config.read_text(encoding='utf-8'))
    for name in ('DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'DEEPSEEK_MODEL'):
        if settings.get(name):
            os.environ.setdefault(name, settings[name])
os.environ.setdefault('YOLO_CONFIG_DIR', str(RUNTIME))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', str(RUNTIME / 'gradio'))
# Preserve the exact research implementation that produced this checkpoint.
sys.path.insert(0, str(ROOT / 'vendor'))
WEIGHTS = ROOT / 'weights' / 'traffic_best.pt'
OUTPUTS = ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
