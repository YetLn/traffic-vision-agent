"""在科研数据集里找本项目检测器能检出 point-l 的图片，供指路牌文字流程使用。

只读科研工程图片，不修改；结果写入 outputs/pointl_candidates.json，并把命中图片复制到
outputs/pointl_probe/ 便于复用。
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps
from traffic_agent.config import OUTPUTS
from traffic_agent.detector import TrafficSignDetector

SOURCES = [Path(r'D:\images-night\from hainan\imagesnight\val\images'),
           Path(r'D:\images-night\from hainan\imagesnight\test\images')]
LIMIT = 120


def main():
    detector = TrafficSignDetector()
    out_dir = OUTPUTS/'pointl_probe'
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = []
    scanned = 0
    for source in SOURCES:
        if not source.is_dir():
            continue
        for path in sorted(source.glob('*.jpg')):
            if scanned >= LIMIT:
                break
            scanned += 1
            try:
                with Image.open(path) as handle:
                    image = ImageOps.exif_transpose(handle).convert('RGB')
                result = detector.detect(image)
            except Exception:
                continue
            point_l = [d for d in result['detections'] if d['class_id'] == 2]
            if not point_l:
                continue
            best = max(point_l, key=lambda item: item['confidence'])
            target = out_dir/f'pointl_{path.stem}.jpg'
            image.save(target, quality=95)
            hits.append({'source':str(path), 'copy':str(target), 'image_size':list(image.size),
                         'point_l_count':len(point_l), 'best_confidence':best['confidence'],
                         'bbox':best['bbox']})
            print(f"命中 {path.name} point-l×{len(point_l)} 最高{best['confidence']:.3f}")
            if len(hits) >= 8:
                break
        if len(hits) >= 8 or scanned >= LIMIT:
            break
    report = {'scanned':scanned, 'hits':len(hits), 'records':hits}
    (OUTPUTS/'pointl_candidates.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'扫描 {scanned} 张，命中 {len(hits)} 张；报告：{OUTPUTS/"pointl_candidates.json"}')


if __name__ == '__main__':
    main()
