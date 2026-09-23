"""对检出 point-l 的真实图片跑牌面文字提取（本地流程，无 API 用量）。

用法：
    .venv\\Scripts\\python.exe scripts\probe_sign_text.py
    .venv\\Scripts\\python.exe scripts\probe_sign_text.py --live-llm
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps
from traffic_agent.agent import ask
from traffic_agent.config import OUTPUTS
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.sign_text import crop_for, read_signs, summarize


def main():
    detector = TrafficSignDetector()
    records = json.loads((OUTPUTS/'pointl_candidates.json').read_text(encoding='utf-8'))['records']
    out = []
    for record in records[:3]:
        with Image.open(record['copy']) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
        signs = read_signs(image, detector)
        crop_dir = OUTPUTS/'sign_crops'
        crop_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for sign in signs:
            path = crop_dir/f"{Path(record['copy']).stem}_{sign['sign_id']}.jpg"
            sign['crop'].save(path, quality=95)
            saved.append(str(path))
        print(f"=== {Path(record['copy']).name} 检出牌面 {len(signs)} 块")
        for row in summarize(signs):
            print(f"    牌{row['sign_id']} 置信度{row['检测置信度']} 文字："
                  + ' / '.join(row['牌面文字']))
        out.append({'image':record['copy'], 'crops':saved, 'signs':summarize(signs)})
        if '--live-llm' in sys.argv and signs:
            result = ask(detector, image, '这块牌想表达什么？', mode='DeepSeek Tool Calling')
            print(f"    [DeepSeek] 工具={[t['tool'] for t in result['trace']]}")
            print(f"    [DeepSeek] 回答：{result['answer'][:600]}")
            out[-1]['live_llm'] = {'tools':[t['tool'] for t in result['trace']],
                                   'answer':result['answer']}
    (OUTPUTS/'sign_text_probe.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"报告：{OUTPUTS/'sign_text_probe.json'}")


if __name__ == '__main__':
    main()
