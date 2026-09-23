"""评测提示/规则类文本分类，报告精确率、召回率与逐条误判。

用法：
    .venv\\Scripts\\python.exe scripts\\evaluate_prompt_classifier.py
报告写入 outputs/prompt_classifier_eval.json。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import OUTPUTS
from traffic_agent.sign_text import classify_text
from prompt_text_gold import CORPUS


def main():
    rows = []
    counters = {label: {'tp':0, 'fp':0, 'fn':0} for label in ('prompt', 'info', 'place')}
    correct_total = 0
    for text, label in CORPUS:
        kind, matched = classify_text(text)
        correct = kind == label
        correct_total += 1 if correct else 0
        for target in counters:
            if kind == target and label == target:
                counters[target]['tp'] += 1
            elif kind == target and label != target:
                counters[target]['fp'] += 1
            elif kind != target and label == target:
                counters[target]['fn'] += 1
        rows.append({'text':text, 'label':label, 'predicted':kind,
                     'matched':matched, 'correct':correct})
    report = {'samples':len(CORPUS), 'accuracy':round(correct_total/len(CORPUS), 3),
              'per_label':{}, 'rows':rows}
    for label, counter in counters.items():
        precision = counter['tp']/(counter['tp']+counter['fp']) if counter['tp']+counter['fp'] else None
        recall = counter['tp']/(counter['tp']+counter['fn']) if counter['tp']+counter['fn'] else None
        report['per_label'][label] = {
            'total':sum(1 for _, value in CORPUS if value == label),
            'precision':round(precision, 3) if precision is not None else None,
            'recall':round(recall, 3) if recall is not None else None, **counter}
    path = OUTPUTS/'prompt_classifier_eval.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"样本 {report['samples']}  整体准确率 {report['accuracy']}")
    for label, stats in report['per_label'].items():
        print(f"  {label:<6} 共{stats['total']:>3} 精确率 {stats['precision']} "
              f"召回率 {stats['recall']}  TP={stats['tp']} FP={stats['fp']} FN={stats['fn']}")
    for row in rows:
        if not row['correct']:
            print(f"  误判：{row['text']!r} 标注={row['label']} 预测={row['predicted']} "
                  f"命中={row['matched']}")
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
