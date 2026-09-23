"""端到端验证：真实图片 → 检测 point-l → 原图裁剪 → OCR → （可选）真实 DeepSeek 意图推断。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_sign_text.py            # 仅本地流程，无 API 用量
    .venv\\Scripts\\python.exe scripts\\verify_sign_text.py --live-llm # 追加真实 DeepSeek 推断

报告写入 outputs/sign_text_verification.json。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.agent import ask
from traffic_agent.config import OUTPUTS, ROOT
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.sign_text import inference_prompt, read_signs, summarize

IMAGES = ['night_01.jpg', 'night_02.jpg', 'night_03.jpg']


def main():
    detector = TrafficSignDetector()
    records = []
    with Image.open(ROOT/'test_images'/IMAGES[0]) as first:
        image = first.convert('RGB')
    signs = read_signs(image, detector)
    records.append({'image':IMAGES[0], 'sign_count':len(signs),
                    'signs':summarize(signs),
                    'prompt':inference_prompt(signs)})

    # 真实 DeepSeek：问“这块牌想表达什么”，检查是否调用 read_sign_text 且标注推测
    live = None
    if '--live-llm' in sys.argv:
        result = ask(detector, image, '这块牌想表达什么？', mode='DeepSeek Tool Calling')
        tools_used = [row['tool'] for row in result['trace']]
        live = {'answer':result['answer'], 'tools':tools_used,
                'called_read_sign_text':'read_sign_text' in tools_used,
                'marked_as_inference':('推测' in result['answer']) or ('推断' in result['answer'])}
        assert live['called_read_sign_text'], '真实 LLM 未调用 read_sign_text'
        assert live['marked_as_inference'], '回答未标注为推测'

    report = {'images_checked':[IMAGES[0]], 'local_runs':records, 'live_llm':live,
              'note':'牌面文字为 OCR 原文；意图为模型推测。测试图片为夜间样例，可能未检出 point-l。'}
    path = OUTPUTS/'sign_text_verification.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2)[:3000])
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
