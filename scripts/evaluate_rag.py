"""Development-only retrieval check; no API or private images are used."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent import rag
from traffic_agent.config import ROOT


def main():
    cases = json.loads((ROOT / 'knowledge/rag_eval.development.json').read_text(encoding='utf-8'))
    rows = []
    for case in cases['cases']:
        result = rag.search(case['question'])
        if case.get('unanswerable'):
            passed = not result['available']
        else:
            passed = any(hit['title'] in case['titles'] for hit in result['hits'])
        rows.append({'question': case['question'], 'unanswerable': bool(case.get('unanswerable')),
                     'passed': passed, 'retrieved': [hit['title'] for hit in result['hits']],
                     'scores': [hit['score'] for hit in result['hits']]})
    answerable = [row for row in rows if not row['unanswerable']]
    negative = [row for row in rows if row['unanswerable']]
    report = {'evaluation_scope': cases['scope'], 'human_verified': False,
              'retriever': 'bm25_chinese_char_ngrams', 'corpus_size': len(rag.corpus()),
              'positive_top3_hits': sum(row['passed'] for row in answerable),
              'positive_total': len(answerable),
              'negative_correct_abstentions': sum(row['passed'] for row in negative),
              'negative_total': len(negative), 'cases': rows}
    path = ROOT / 'outputs/rag_stage1_evaluation.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'},
                     ensure_ascii=False, indent=2))
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
