"""真实 DeepSeek 多轮压缩验收：只发送文字与结构化统计，不发送图片或本机路径。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_context_compression.py

默认使用人工构造的检测结果与内存图片，不读取本机图片、不加载 YOLO 权重。
会产生 DeepSeek API 用量。报告写入 outputs/context_compression.json。
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.agent import ask, summary_note
from traffic_agent.config import ROOT

QUESTIONS = [
    '有多少个 ban？',
    '把它们框出来。',
    '哪个置信度最高？',
    '图里有什么交通标志？',
    '有多少个 point-l？',
    '再框一次 ban。',
    '总数是多少？',
    '之前的对话摘要是什么？',
]


class SyntheticDetector:
    """人工构造的检测结果：不触发本机推理，也不产生真实图片内容。"""

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
    records, state = [], None
    with tempfile.TemporaryDirectory() as directory, Image.new('RGB', (10, 10)) as blank:
        detector = SyntheticDetector(directory)
        for index, question in enumerate(QUESTIONS, start=1):
            result = ask(detector, blank, question, mode='DeepSeek Tool Calling', state=state)
            state = result['state']
            records.append({'round':index, 'question':question, 'answer':result['answer'],
                            'tools':[t['tool'] for t in result['trace']],
                            'folded_turns':state['context']['folded_turns'],
                            'history_messages':len(state['history'])})
        folded = [r for r in records if r['folded_turns']]
        assert folded, '压缩未触发，请检查 MAX_MESSAGES 设置'
        assert records[-1]['history_messages'] <= 6, '历史未被压住'
        # 离线规则模式必须直接用折叠上下文回答，且不调用工具。
        offline_state = None
        for _ in range(4):
            offline_state = ask(detector, blank, '图里有什么交通标志？', state=offline_state)['state']
        offline = ask(detector, blank, '之前的对话摘要是什么？', state=offline_state)
        assert offline['trace'] == [] and '已折叠的会话上下文' in offline['answer'], offline['answer']
    note = summary_note(state)
    assert 'ban' in note and 'point-l' not in note, note
    report = {
        'provider':'DeepSeek',
        'payload':'仅问题、折叠后的文字上下文与结构化检测统计；未发送图片像素或本机绝对路径',
        'data_source':'人工构造检测结果 + 内存空白图；未读取 test_images 或 weights',
        'compression':{'max_messages':6, 'kept_history_messages':records[-1]['history_messages'],
                       'folded_turns':records[-1]['folded_turns'],
                       'summary':json.loads(note.split('（JSON）：', 1)[-1])},
        'rounds':records,
    }
    path = ROOT/'outputs/context_compression.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'报告：{path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'failed', 'error_type':type(exc).__name__,
                          'http_status':getattr(exc, 'status_code', None)}, ensure_ascii=False))
        sys.exit(1)
