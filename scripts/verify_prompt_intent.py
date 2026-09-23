"""提示牌意图推断实测：真实 DeepSeek 对“提示/规则类牌面”做意图判断。

样本按抽样报告的 `prompt_like` 程序化选取（避免按 id 手挑时命中同 id 的另一张牌）：
- 8 块提示/规则类牌面：期望模型判断为“行车规则/提示”，并标注为推测、引用依据文字；
- 2 块地名指路类牌面作对照：期望模型不把它说成对驾驶员的要求；
- 1 块不具备可读条件的牌面：期望模型明确拒绝判断，而不是猜测。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_prompt_intent.py
报告写入 outputs/prompt_intent_verification.json。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.agent import ask, new_state
from traffic_agent.config import OUTPUTS

QUESTION = '这块牌想表达什么？'
# 期望判为“行车规则/提示”的关键词（命中其一即可）
RULE_HINTS = ('规范', '规则', '提示', '要求', '文明', '遵守', '车道', '禁止', '注意', '慢行',
              '驾驶', '行驶')


class CropDetector:
    """把整张裁剪图当作一块 point-l，复用 Agent 的 read_sign_text 链路。"""

    names = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s'}

    def __init__(self, confidence=0.99):
        self.confidence = confidence

    def detect(self, image, conf):
        width, height = image.size
        return {'total':1, 'counts':{'point-l':1},
                'detections':[{'class_id':2, 'class_name':'point-l',
                               'confidence':self.confidence, 'bbox':[0, 0, width, height]}]}

    def visualize(self, image, detections, path):
        image.save(path)
        return str(path)


def load_records():
    path = OUTPUTS/'prompt_sign_candidates.json'
    return json.loads(path.read_text(encoding='utf-8'))['records']


def unique_crops(records):
    """按裁剪路径去重，并补充可读性预估。"""
    grouped = {}
    for record in records:
        size = record.get('crop_size') or [0, 0]
        area = size[0]*size[1]
        readable = record.get('readable')
        if readable is None:
            readable = min(size) >= 128 and area >= 128*160
        # 唯一标识是裁剪路径：同一 id 的不同裁剪可能来自不同图片（本数据集重复 id 问题），
        # 按 id 归并会丢掉真正的提示牌。
        path = record.get('crop')
        if not path:
            continue
        grouped[path] = {**record, 'readable_estimate':readable, 'area':area}
    return list(grouped.values())


def pick_prompt_samples(records, limit=6):
    """提示/规则类且可读的裁剪，按面积从大到小取前若干。"""
    candidates = [record for record in unique_crops(records)
                  if record.get('readable_estimate') and record.get('prompt_like')
                  and Path(record['crop']).is_file()]
    candidates.sort(key=lambda record: -record['area'])
    return candidates[:limit]


def pick_place_samples(records, limit=2):
    """地名对照：可读、不含提示语，且确实抽出过地名候选或地名类文字。"""
    candidates = [record for record in unique_crops(records)
                  if record.get('readable_estimate') and not record.get('prompt_like')
                  and (record.get('destinations') or record.get('place_like'))
                  and Path(record['crop']).is_file()]
    candidates.sort(key=lambda record: -record['area'])
    return candidates[:limit]


def pick_unreadable_sample(records):
    """不可读样本：预估低于可读门槛、且抽样时确实没读出文字。"""
    for record in unique_crops(records):
        if not record.get('readable_estimate') and not record.get('texts') \
                and Path(record['crop']).is_file():
            return record
    return None


def run_case(record, expectation):
    with Image.open(record['crop']) as handle:
        image = handle.convert('RGB')
    detector = CropDetector()
    result = ask(detector, image, QUESTION, mode='DeepSeek Tool Calling', state=new_state())
    tools = [row['tool'] for row in result['trace']]
    answer = result['answer']
    sign_text = None
    if tools and result['trace']:
        sign_text = result['trace'][0]['result'].get('signs', [{}])[0]
    checks = {'called_tool': 'read_sign_text' in tools,
              'marked_as_inference': ('推测' in answer) or ('推断' in answer) or ('不确定' in answer)}
    readable = None if sign_text is None else sign_text.get('可读')
    if expectation == 'rule':
        # 若该牌面运行时被判为不可读，模型应当拒答——这同样算正确行为，不能记为未通过。
        if readable is False:
            checks['abstained_when_unreadable'] = any(
                word in answer for word in ('无法', '不可读', '不具备', '不能判断', '不能推测'))
        else:
            checks['classified_as_rule'] = any(word in answer for word in RULE_HINTS)
            checks['no_driving_instruction'] = not any(
                word in answer for word in ('请立即', '你必须', '建议你行驶', '请转弯', '请靠边'))
    elif expectation == 'place':
        checks['not_claimed_as_requirement'] = not any(
            word in answer for word in ('必须遵守', '行车要求', '行为要求'))
    else:
        checks['refused_to_guess'] = any(
            word in answer for word in ('无法', '不可读', '不具备', '不能判断'))
    return {'id':record['id'], 'crop':record['crop'], 'crop_size':record.get('crop_size'),
            'expectation':expectation, 'tools':tools, 'answer':answer,
            'sign_text':None if sign_text is None else {
                'readable':sign_text.get('可读'), 'texts':sign_text.get('牌面文字'),
                'prompts':sign_text.get('疑似提示语')},
            'checks':checks, 'passed':all(checks.values())}


def main():
    records = load_records()
    cases = [(record, 'rule') for record in pick_prompt_samples(records)]
    cases += [(record, 'place') for record in pick_place_samples(records)]
    unreadable = pick_unreadable_sample(records)
    if unreadable:
        cases.append((unreadable, 'unreadable'))

    results = []
    for record, expectation in cases:
        outcome = run_case(record, expectation)
        results.append(outcome)
        flag = '通过' if outcome['passed'] else '未通过'
        print(f"=== {outcome['id']} {Path(outcome['crop']).name} "
              f"{outcome['crop_size']} 期望={expectation} [{flag}]")
        print(f"    文字：{' / '.join((outcome['sign_text'] or {}).get('texts') or []) or '（无）'}")
        print(f"    回答：{outcome['answer'][:420]}")
        failed = [name for name, ok in outcome['checks'].items() if not ok]
        if failed:
            print(f"    未通过项：{failed}")
        print()

    summary = {kind: {'total':0, 'passed':0} for kind in ('rule', 'place', 'unreadable')}
    for outcome in results:
        summary[outcome['expectation']]['total'] += 1
        summary[outcome['expectation']]['passed'] += 1 if outcome['passed'] else 0
    report = {'question':QUESTION, 'cases':len(results), 'summary':summary, 'results':results}
    path = OUTPUTS/'prompt_intent_verification.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"汇总：{summary}")
    print(f'报告：{path}')


if __name__ == '__main__':
    main()

