"""提示/规则类牌面的端到端验证：从裁出的牌面到文字分类与分组结果。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_prompt_signs.py
报告写入 outputs/prompt_signs_verification.json。

注意：本数据集不同批次目录存在**重复 id 且为不同图片**，因此案例按“id + 期望文字”
从抽样报告里定位裁剪路径；只按 id 会取到另一张同 id 牌面（已实测踩坑）。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.agent import new_state
from traffic_agent.config import OUTPUTS
from traffic_agent.sign_text import classify_text
from traffic_agent.tools import ToolContext

# (id, 期望出现的提示/规则类文字)
CASES = [
    ('08327', '请按规定车道行驶'),
    ('08325', '各行其道'),
    ('08525', '请按导向车道行驶'),
    ('08504', '请选定行车方向'),
    ('08441', '前方路口禁止左转'),
    ('08324', '请选择车道行驶'),
    ('08086', '请止辅道'),
    ('08274', '禁止通行'),
]


class CropDetector:
    """把整张裁剪图当作一块 point-l，复用 Agent 的 read_sign_text 链路。"""

    names = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s'}

    def detect(self, image, conf):
        width, height = image.size
        return {'total':1, 'counts':{'point-l':1},
                'detections':[{'class_id':2, 'class_name':'point-l', 'confidence':0.99,
                               'bbox':[0, 0, width, height]}]}

    def visualize(self, image, detections, path):
        image.save(path)
        return str(path)


def load_hits():
    """从抽样报告的提示类记录里取 (id, 提示文字列表, 裁剪路径)。"""
    path = OUTPUTS/'prompt_sign_candidates.json'
    hits = []
    for record in json.loads(path.read_text(encoding='utf-8'))['records']:
        if record.get('prompt_like'):
            hits.append((record['id'], record['prompt_like'], Path(record['crop'])))
    return hits


def resolve(sign_id, needle):
    for record_id, prompts, crop in load_hits():
        if record_id == sign_id and any(needle in text for text in prompts) and crop.is_file():
            return crop
    return None


def main():
    records = []
    for sign_id, needle in CASES:
        path = resolve(sign_id, needle)
        if path is None:
            print(f'{sign_id}（{needle}）：找不到对应裁剪，跳过')
            continue
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        context = ToolContext(CropDetector(), image, .25, new_state())
        output = context.execute('read_sign_text', {})
        row = output['signs'][0]
        hit = any(needle in text for text in row['疑似提示语'])
        records.append({'id':sign_id, 'crop':str(path), 'crop_size':list(image.size),
                        'expected':needle, 'hit':hit, 'readable':row['可读'],
                        'texts':row['牌面文字'], 'prompts':row['疑似提示语'],
                        'destinations':row['地名候选'],
                        'low_confidence':row['低置信度文字'],
                        'types':[{'text':text, 'type':classify_text(text)[0],
                                  'matched':classify_text(text)[1]}
                                 for text in row['牌面文字']]})
        print(f"=== {sign_id} {path.name} {image.size} 可读={row['可读']} "
              f"期望『{needle}』{'命中' if hit else '未命中'}")
        print(f"    文字：{' / '.join(row['牌面文字']) or '（无）'}")
        print(f"    提示语：{' / '.join(row['疑似提示语']) or '（无）'}   "
              f"地名候选：{' / '.join(row['地名候选']) or '（无）'}")
    report = {'cases':len(records), 'hits':sum(1 for record in records if record['hit']),
              'records':records}
    out = OUTPUTS/'prompt_signs_verification.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n{len(records)} 块牌面中 {report['hits']} 块命中期望的提示/规则类文字")
    print(f'报告：{out}')


if __name__ == '__main__':
    main()
