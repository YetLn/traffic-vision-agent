"""指路牌解析评测：用人工 ground truth 量化三项指标。

指标定义（以“地名”为主键，而不是以箭头为主键，因为业务真正关心的是地名对不对）：
- 文字识别正确率：ground truth 里可见地名中，被 OCR 文本按“去空格后包含”命中的比例；
- 方向正确率：ground truth 中可判定的 route，被解析结果同名地名给出正确方向的比例；
- 关联正确率（严格）：ground truth 中可判定的 route，被解析结果**同名且方向一致**命中的比例；
- 过度断言：ground truth 标为歧义/应拒答的地名，被解析器强行给出方向的数量（越低越好）。

用法：
    .venv\\Scripts\\python.exe scripts\\evaluate_sign_parser.py
    .venv\\Scripts\\python.exe scripts\\evaluate_sign_parser.py --samples 路径 --output 报告.json
报告写入 outputs/sign_parse_eval.json。
所有样本必须在推理前通过校验；缺文件、损坏图片或重复 ID 会终止评测，保留原报告。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.config import OUTPUTS
from traffic_agent.ocr import read_text
from traffic_agent.sign_parser import parse_sign
from scripts.eval_ground_truth import EVAL_SET

SAMPLES = OUTPUTS/'ocr_samples'


def normalize(text):
    return ''.join(str(text).split()).lower()


def validate_cases(cases, samples):
    """一次性检查完整评测输入，禁止因样本缺失而悄悄改变评测分母。"""
    errors, seen, validated = [], set(), []
    if not cases:
        errors.append('评测集为空')
    for index, case in enumerate(cases, 1):
        case_id = case.get('id')
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f'第 {index} 条样本缺少非空字符串 ID')
        elif case_id.strip() in seen:
            errors.append(f'重复样本 ID：{case_id}')
        else:
            seen.add(case_id.strip())
        label = case_id or f'第 {index} 条'
        crop = case.get('crop')
        if not isinstance(crop, str) or not crop.strip():
            errors.append(f'{label}：缺少 crop 路径')
            continue
        path = samples / crop
        if not path.is_file():
            errors.append(f'{label}：缺少图片 {path}')
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f'{label}：图片无法读取 {path}（{exc}）')
            continue
        validated.append((case, path))
    if errors:
        raise ValueError('评测输入校验失败，未运行 OCR，原报告保持不变：\n' + '\n'.join(errors))
    return validated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=SAMPLES, help='人工 gold 中 crop 相对路径所在目录')
    parser.add_argument('--output', type=Path, default=OUTPUTS/'sign_parse_eval.json', help='评测报告 JSON 路径')
    args = parser.parse_args(argv)
    try:
        cases = validate_cases(EVAL_SET, args.samples)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows, totals = [], {'gold_text':0, 'hit_text':0, 'gold_route':0, 'correct_route':0,
                        'linked_route':0, 'overclaim':0, 'ambiguous':0}
    for case, path in cases:
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        ocr = read_text(image)
        ocr_text = normalize(' '.join(line['text'] for line in ocr['lines']))
        parsed = parse_sign(image)

        gold_destinations = [route['destination'] for route in case['routes']]
        hit = [name for name in gold_destinations if normalize(name) in ocr_text]
        got = {route['destination']:route['direction'] for route in parsed['routes']}
        # 解析出的地名可能是“目的地组”拼接串，用包含关系匹配
        def lookup(name):
            for key, direction in got.items():
                if normalize(name) in normalize(key):
                    return direction
            return None
        correct = sum(1 for route in case['routes'] if lookup(route['destination']) == route['direction'])
        linked = sum(1 for route in case['routes'] if lookup(route['destination']) is not None)
        overclaim = sum(1 for name in case['ambiguous'] if lookup(name) is not None)

        totals['gold_text'] += len(gold_destinations)
        totals['hit_text'] += len(hit)
        totals['gold_route'] += len(case['routes'])
        totals['correct_route'] += correct
        totals['linked_route'] += linked
        totals['ambiguous'] += len(case['ambiguous'])
        totals['overclaim'] += overclaim
        rows.append({'id':case['id'], 'difficulty':case['difficulty'],
                     'gold_routes':case['routes'], 'parsed_routes':parsed['routes'],
                     'gold_text_hit':[f'{name}:{"命中" if name in hit else "未命中"}'
                                      for name in gold_destinations],
                     'route_correct':correct, 'route_linked':linked, 'overclaim':overclaim,
                     'ocr_text':[line['text'] for line in ocr['lines']],
                     'notes':case['notes']})
        print(f"{case['id']:>6} {case['difficulty']:<14} gold路线{len(case['routes'])} "
              f"命中{correct} 关联{linked} 过度断言{overclaim} "
              f"| 解析: {'; '.join(f'{k}→{v}' for k, v in got.items())[:60] or '无'}")

    def ratio(hit, total):
        return round(hit/total, 3) if total else None
    report = {
        'samples':len(rows),
        'text_recall':ratio(totals['hit_text'], totals['gold_text']),
        'direction_accuracy_of_linked':ratio(totals['correct_route'], totals['linked_route']),
        'association_accuracy_strict':ratio(totals['correct_route'], totals['gold_route']),
        'link_rate':ratio(totals['linked_route'], totals['gold_route']),
        'overclaim_on_ambiguous':totals['overclaim'],
        'ambiguous_total':totals['ambiguous'],
        'totals':totals, 'rows':rows,
    }
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n样本 {report['samples']} 块")
    print(f"文字召回        {report['text_recall']}")
    print(f"已关联方向正确率 {report['direction_accuracy_of_linked']}")
    print(f"关联严格正确率   {report['association_accuracy_strict']}")
    print(f"关联覆盖率       {report['link_rate']}")
    print(f"歧义地名被强行断言 {report['overclaim_on_ambiguous']}/{report['ambiguous_total']}")
    print(f'报告：{path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
