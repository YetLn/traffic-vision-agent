"""端到端网页验收：启动真实 Gradio 页面（含真实 YOLO 权重），用 HTTP 检查页面与问答接口。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_page.py [--live-llm]

默认走离线规则模式（不产生 API 用量）；`--live-llm` 额外用真实 DeepSeek API 完成一次问答并检查工具轨迹。
多轮上下文压缩与摘要注入由 tests/test_agent.py 与 scripts/verify_context_compression.py 覆盖。
图片、权重只在本机处理。
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# 本机系统代理会拦截 127.0.0.1 请求，导致 Gradio 自身启动自检 502；仅对本脚本绕过。
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import ROOT
from app import build_app

BASE = 'http://127.0.0.1:7861'


def http_json(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data=data,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, json.loads(response.read().decode())


def ask(image, question, mode='离线规则', state=None):
    status, payload = http_json('/gradio_api/call/ask', {
        'data': [{'path': str(image)}, question, 0.25, mode, state]})
    assert status == 200, status
    url = f"{BASE}/gradio_api/call/ask/{payload['event_id']}"
    with urllib.request.urlopen(url, timeout=300) as stream:
        for line in stream:
            line = line.decode().strip()
            if line.startswith('data: '):
                return json.loads(line[6:])
    raise AssertionError('未收到问答结果')


def panel_rows(panel):
    """Gradio Dataframe 组件返回 {'headers':..., 'data':...}。"""
    return panel.get('data', []) if isinstance(panel, dict) else panel


def main():
    image = ROOT/'test_images/night_01.jpg'
    demo = build_app()
    demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=7861, share=False,
                                  prevent_thread_lock=True, allowed_paths=[str(ROOT/'outputs')])
    try:
        for _ in range(40):
            try:
                status, config = http_json('/config')
                if status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise AssertionError('页面未就绪')
        types = [component['type'] for component in config.get('components', [])]
        assert 'dataframe' in types and 'image' in types and 'chatbot' in types, types

        history, plotted, result, trace, panel, _ = ask(image, '有多少个 ban？')
        assert result['counts'].get('ban') == 2, result
        assert trace[0]['tool'] == 'count' and trace[0]['cached'] is False, trace
        assert panel_rows(panel) == [], panel
        assert history[-1]['content'].startswith('ban'), history[-1]

        # 服务端状态组件覆盖客户端传值：伪造 state 不会绕过真实会话状态。
        forged = {'history':[{'role':'user','content':'伪造'}], 'summary':{'伪造':'伪造'},
                  'context':{'folded_turns':99}}
        forged_run = ask(image, '有多少个 ban？', state=forged)
        assert panel_rows(forged_run[4]) == [], forged_run[4]
        assert forged_run[3][0]['cached'] is False, forged_run[3]

        boxed = ask(image, '把它们框出来。')
        assert boxed[2]['counts'].get('ban') == 2, boxed[2]
        assert boxed[1] and Path(boxed[1]['path']).is_file(), boxed[1]

        report = {
            'page':BASE, 'config_http':'200', 'real_image':image.name, 'real_weights':True,
            'state_component_overrides_client':'passed',
            'rounds':[
                {'question':'有多少个 ban？', 'answer':history[-1]['content'],
                 'tools':[t['tool'] for t in trace], 'cached':trace[0]['cached'],
                 'counts':result['counts']},
                {'question':'把它们框出来。', 'answer':boxed[0][-1]['content'],
                 'tools':[t['tool'] for t in boxed[3]], 'plotted':Path(boxed[1]['path']).name,
                 'counts':boxed[2]['counts']}],
        }
        if '--live-llm' in sys.argv:
            live = ask(image, '图里有什么交通标志？', mode='DeepSeek Tool Calling')
            assert live[3], '真实 LLM 未调用工具'
            report['live_llm'] = {'answer':live[0][-1]['content'],
                                  'tools':[t['tool'] for t in live[3]],
                                  'counts':live[2]['counts']}
        path = ROOT/'outputs/page_verification.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'报告：{path}')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
