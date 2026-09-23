"""对比箭头检测两种方法，并对已知样本核对方向。

用法：
    .venv\\Scripts\\python.exe scripts\\compare_arrow_methods.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.arrows import detect_arrows, detect_arrows_template
from traffic_agent.config import OUTPUTS

# 人工看图得出的参考（仅这三块用于核对方法是否可用，不作为完整评测集）
CASES = ['08342_1.jpg', '08342_2.jpg', '08118_2.jpg', '08845_1.jpg', '08419_1.jpg']


def main():
    for name in CASES:
        path = OUTPUTS/'ocr_samples'/name
        if not path.is_file():
            print(f'{name}: 缺文件')
            continue
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        template = detect_arrows_template(image)
        geometric = detect_arrows(image)
        t_summary = ', '.join(f"{a['direction']}({a['score']})" for a in template['arrows'])
        g_count = len(geometric['arrows'])
        print(f"--- {name} {image.size}")
        print(f"    模板法: {len(template['arrows'])} 个 (候选 {template['candidates']})  {t_summary}")
        print(f"    几何法: {g_count} 个")


if __name__ == '__main__':
    main()
