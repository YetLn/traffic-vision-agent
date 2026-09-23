"""在真实指路牌裁剪图上验证 OCR：必须同时拿到文字与坐标。

用法：
    .venv\\Scripts\\python.exe scripts\\probe_ocr.py                # 跑清单里前若干张
    .venv\\Scripts\\python.exe scripts\\probe_ocr.py --limit 12
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import OUTPUTS

MANIFEST = OUTPUTS/'ocr_samples.json'


def main():
    parser = argparse.ArgumentParser(description='RapidOCR 真实牌面验证')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--ids', default='', help='逗号分隔的样本 id，例如 08342,08118')
    args = parser.parse_args()

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    crops = json.loads(MANIFEST.read_text(encoding='utf-8'))['crops']
    if args.ids:
        wanted = set(args.ids.split(','))
        crops = [c for c in crops if c['id'] in wanted][:args.limit]
    else:
        crops = crops[:args.limit]

    results, texts_total = [], 0
    for item in crops:
        start = time.perf_counter()
        raw, _ = engine(item['crop'])
        elapsed = round((time.perf_counter()-start)*1000, 1)
        lines = []
        for row in (raw or []):
            box, text, score = row[0], row[1], float(row[2])
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            lines.append({'text':text, 'score':round(score, 3),
                          'bbox':[round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]})
        lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
        texts_total += len(lines)
        results.append({'id':item['id'], 'crop':item['crop'], 'crop_size':item['crop_size'],
                        'elapsed_ms':elapsed, 'text_count':len(lines), 'lines':lines})
        print(f"--- {item['id']} ({item['crop_size'][0]}x{item['crop_size'][1]}) "
              f"{len(lines)} 段文字 {elapsed} ms")
        for line in lines:
            print(f"    {line['text']!r} score={line['score']} bbox={line['bbox']}")

    report = {'engine':'rapidocr_onnxruntime', 'crops_tested':len(results),
              'text_lines_total':texts_total,
              'avg_ms':round(sum(r['elapsed_ms'] for r in results)/max(1, len(results)), 1),
              'results':results}
    path = OUTPUTS/'ocr_probe.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n平均耗时 {report["avg_ms"]} ms，共 {texts_total} 段文字；报告：{path}')


if __name__ == '__main__':
    main()
