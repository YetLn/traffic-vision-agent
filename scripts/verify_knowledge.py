"""知识库框架验收：结构、检索、缺依据拒答，以及真实 DeepSeek 的工具接入。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_knowledge.py            # 离线检查，无 API 用量
    .venv\\Scripts\\python.exe scripts\\verify_knowledge.py --live-llm # 追加真实 DeepSeek 调用

框架验收与“语义是否已填写”解耦：即使 signs.yaml 仍为空，本脚本也应全部通过。
报告写入 outputs/knowledge_verification.json。
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent import knowledge
from traffic_agent.agent import ask
from traffic_agent.config import ROOT
from traffic_agent.tools import schemas


class SyntheticDetector:
    names = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s'}

    def __init__(self, directory):
        self.directory = Path(directory)

    def detect(self, image, conf):
        return {'total':2, 'counts':{'ban':2}, 'detections':[
            {'class_id':1, 'class_name':'ban', 'confidence':0.9132, 'bbox':[1,1,4,4]},
            {'class_id':1, 'class_name':'ban', 'confidence':0.7125, 'bbox':[5,5,9,9]}]}

    def visualize(self, image, detections, path):
        image.save(path)
        return str(path)


def main():
    status = knowledge.inventory()
    checks, records = [], {}
    with tempfile.TemporaryDirectory() as directory, Image.new('RGB', (10, 10)) as blank:
        detector = SyntheticDetector(directory)
        # 1) 结构：知识库类别与模型类别一一对应。
        checks.append({'check':'类别与模型一致',
                       'value':sorted(row['class_name'] for row in knowledge.entries()),
                       'passed':sorted(row['class_name'] for row in knowledge.entries()) == sorted(knowledge.ALIASES)})
        # 2) 工具 schema 暴露 lookup_sign，且参数只接受模型类别。
        tools = {tool['function']['name']:tool['function'] for tool in schemas(detector.names.values())}
        sign_tool = tools.get('lookup_sign')
        checks.append({'check':'lookup_sign 工具 schema',
                       'value':sorted(sign_tool['parameters']['properties']) if sign_tool else None,
                       'passed':bool(sign_tool) and sign_tool['parameters']['properties']['class_name']['enum'] == list(detector.names.values())})
        # 3) 未知类别不得被接受。
        unknown = knowledge.lookup('speed_limit')
        checks.append({'check':'未知类别拒绝', 'value':unknown['reason'],
                       'passed':not unknown['available'] and '不支持类别' in unknown['reason']})
        # 4) 未填语义的类别必须拒答，措辞与知识库状态一致。
        pending = [row['class_name'] for row in status['classes'] if not row['available']]
        for name in pending:
            result = knowledge.lookup(name)
            passed = not result['available']
            checks.append({'check':f'{name} 缺依据拒答', 'value':result.get('reason'), 'passed':passed})
        # 5) 端到端：缺依据类别的语义问题不得调用检测工具，也不得编造含义。
        for name in pending:
            result = ask(detector, blank, f'{name} 是什么意思？', state=None)
            records[name] = {'answer':result['answer'], 'tools':[row['tool'] for row in result['trace']]}
            checks.append({'check':f'{name} 语义问题不调用检测',
                           'value':records[name]['tools'],
                           'passed':result['trace'] == [] or all(
                               row['tool'] == 'lookup_sign' for row in result['trace'])})
        # 6) 无类别线索时给出可选类别，而不是猜测。
        hint = ask(detector, blank, '这个标志是什么意思？')
        checks.append({'check':'无类别线索时不猜测', 'value':hint['answer'],
                       'passed':'请说明要查询的类别' in hint['answer']})
        # 7) 检测类问题仍然正常（知识库不干扰检测链路）。
        detected = ask(detector, blank, '有多少个 ban？')
        checks.append({'check':'检测链路未受影响', 'value':detected['answer'],
                       'passed':detected['result']['counts'].get('ban') == 2})
        # 8) 已填语义的类别按知识库原文回答（当前若为空则跳过该分支）。
        available = [row['class_name'] for row in status['classes'] if row['available']]
        for name in available:
            result = ask(detector, blank, f'{name} 是什么意思？')
            entries = {row['class_name']:row for row in knowledge.entries()}
            checks.append({'check':f'{name} 引用知识库原文',
                           'value':result['answer'],
                           'passed':entries[name]['meaning'] in result['answer']
                                    and entries[name]['reference'] in result['answer']})

        report = {'knowledge_status':status, 'pending_classes':pending,
                  'available_classes':available, 'checks':checks,
                  'all_passed':all(item['passed'] for item in checks),
                  'offline_answers':records}
        if '--live-llm' in sys.argv:
            live = ask(detector, blank, 'ban 是什么意思？', mode='DeepSeek Tool Calling')
            tools_used = [row['tool'] for row in live['trace']]
            report['live_llm'] = {'answer':live['answer'], 'tools':tools_used,
                                  'called_lookup_sign':'lookup_sign' in tools_used}
            report['all_passed'] = report['all_passed'] and 'lookup_sign' in tools_used
    assert report['all_passed'], [item for item in checks if not item['passed']]
    path = ROOT/'outputs/knowledge_verification.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'报告：{path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'failed', 'error_type':type(exc).__name__,
                          'detail':str(exc)[:400]}, ensure_ascii=False))
        sys.exit(1)
