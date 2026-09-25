"""Check the public-document RAG path; --live-llm sends only public snippets."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent import rag
from traffic_agent.config import ROOT


def main():
    questions = [
        '方向解析为什么还没有通过冻结测试？',
        'DeepSeek 会收到上传图片的像素吗？',
        '项目的正式车规认证编号是什么？',
    ]
    mode = 'DeepSeek 引用回答' if '--live-llm' in sys.argv else '离线证据'
    results = [{'question': question, **rag.answer(question, mode)} for question in questions]
    report = {'mode': mode, 'public_sources_only': True, 'questions': results,
              'citation_gate_passed': all(
                  result['answer'] == rag.NO_EVIDENCE or
                  all(item in {hit['id'] for hit in result['hits']}
                      for item in result['used_citations'])
                  for result in results)}
    path = ROOT / ('outputs/rag_live_verification.json' if '--live-llm' in sys.argv
                   else 'outputs/rag_offline_verification.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'mode': mode, 'citation_gate_passed': report['citation_gate_passed'],
                      'answers': [{'question': row['question'], 'answer': row['answer'],
                                   'citations': row['used_citations']} for row in results]},
                     ensure_ascii=False, indent=2))
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
