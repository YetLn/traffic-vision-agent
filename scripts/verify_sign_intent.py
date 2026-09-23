"""高分辨率真实牌面验证：文字提取 + 真实 DeepSeek 意图推断。

使用 outputs/ocr_samples 里从私有数据集按原图分辨率裁出的牌面（大牌、清晰），
这是本功能的目标场景；`scripts/find_pointl_images.py` 已证明夜间远处的牌面不具备可读条件。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_sign_intent.py
    .venv\\Scripts\\python.exe scripts\\verify_sign_intent.py --names 08342_2.jpg,08845_1.jpg --live-llm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.agent import ask
from traffic_agent.config import OUTPUTS
from traffic_agent.ocr import read_text
from traffic_agent.sign_text import split_lines, summarize


class CropDetector:
    """把整张裁剪图当作一块 point-l，复用 Agent 的 read_sign_text 链路。"""

    names = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s'}

    def __init__(self, image):
        self.image = image

    def detect(self, image, conf):
        width, height = image.size
        return {'total':1, 'counts':{'point-l':1},
                'detections':[{'class_id':2, 'class_name':'point-l', 'confidence':0.99,
                               'bbox':[0, 0, width, height]}]}

    def visualize(self, image, detections, path):
        image.save(path)
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', default='08342_2.jpg,08845_1.jpg,08793_1.jpg')
    parser.add_argument('--live-llm', action='store_true')
    args = parser.parse_args()
    records = []
    for name in args.names.split(','):
        path = OUTPUTS/'ocr_samples'/name
        if not path.is_file():
            print(f'缺文件：{path}')
            continue
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        detector = CropDetector(image)
        result = ask(detector, image, '这块牌想表达什么？', mode='离线规则')
        sign_text = result['trace'][0]['result'] if result['trace'] else {}
        print(f"=== {name} {image.size}")
        print(f"    离线抽取：{result['answer'][:150]}")
        record = {'name':name, 'crop_size':list(image.size),
                  'texts':sign_text.get('signs', [{}])[0].get('牌面文字', []),
                  'prompts':sign_text.get('signs', [{}])[0].get('疑似提示语', []),
                  'offline_answer':result['answer']}
        if args.live_llm:
            live = ask(detector, image, '这块牌想表达什么？', mode='DeepSeek Tool Calling')
            record['live_tools'] = [row['tool'] for row in live['trace']]
            record['live_answer'] = live['answer']
            print(f"    [DeepSeek] 工具={record['live_tools']}")
            print(f"    [DeepSeek] 回答：{live['answer'][:700]}")
        records.append(record)
    path = OUTPUTS/'sign_intent_verification.json'
    path.write_text(json.dumps({'records':records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
