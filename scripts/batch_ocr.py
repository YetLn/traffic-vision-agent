"""对 outputs/ocr_samples 里的指路牌裁剪图批量跑 OCR，统计可读性与耗时。

用法：
    .venv\\Scripts\\python.exe scripts\\batch_ocr.py            # 全部样本
    .venv\\Scripts\\python.exe scripts\\batch_ocr.py --limit 20
结果写入 outputs/ocr_batch.json，便于人工逐块核对与后续评测。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import OUTPUTS
from traffic_agent.ocr import plain_text, read_text

MANIFEST = OUTPUTS/'ocr_samples.json'


def main():
    parser = argparse.ArgumentParser(description='批量 OCR 指路牌裁剪图')
    parser.add_argument('--limit', type=int, default=0, help='0 表示全部')
    parser.add_argument('--min-score', type=float, default=0.5)
    args = parser.parse_args()

    crops = json.loads(MANIFEST.read_text(encoding='utf-8'))['crops']
    if args.limit:
        crops = crops[:args.limit]
    records, fail = [], 0
    for item in crops:
        result = read_text(item['crop'], min_score=args.min_score)
        lines = result['lines']
        fail += 1 if not lines else 0
        records.append({'id':item['id'], 'crop':item['crop'], 'crop_size':item['crop_size'],
                        'text_count':len(lines), 'elapsed_ms':result['elapsed_ms'],
                        'texts':[line['text'] for line in lines],
                        'lines':lines})
        preview = ' | '.join(line['text'] for line in lines[:6])
        print(f"{item['id']:>6} {item['crop_size'][0]:>4}x{item['crop_size'][1]:<4} "
              f"{len(lines):>2}段 {str(result['elapsed_ms']):>6}ms  {preview[:70]}")

    counts = [r['text_count'] for r in records]
    report = {'engine':'rapidocr_onnxruntime', 'crops':len(records), 'empty_result':fail,
              'median_text_count':sorted(counts)[len(counts)//2] if counts else 0,
              'avg_ms':round(sum(r['elapsed_ms'] or 0 for r in records)/max(1, len(records)), 1),
              'records':records}
    path = OUTPUTS/'ocr_batch.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n共 {report['crops']} 块牌面，无文字输出 {fail} 块，"
          f"文字段中位数 {report['median_text_count']}，平均 {report['avg_ms']} ms")
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
