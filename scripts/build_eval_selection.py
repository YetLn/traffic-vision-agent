"""从已裁出的牌面中挑 30 块评测样本，并带上 OCR 初读，便于人工核对成 ground truth。

选择策略：每块牌只取一张裁剪（避免同一路牌的相邻照片同时进评测集），
再按裁剪宽度分层抽样，覆盖小牌到超大牌；输出 output/eval_set.json。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from traffic_agent.config import OUTPUTS
from traffic_agent.ocr import read_text

TARGET = 30


def main():
    crops = json.loads((OUTPUTS/'ocr_samples.json').read_text(encoding='utf-8'))['crops']
    # 每块牌只保留最大的一张裁剪
    best = {}
    for item in crops:
        if item['id'] not in best or item['crop_size'][0] > best[item['id']]['crop_size'][0]:
            best[item['id']] = item
    items = sorted(best.values(), key=lambda item: item['crop_size'][0])
    if len(items) <= TARGET:
        chosen = items
    else:
        step = (len(items)-1)/(TARGET-1)
        chosen = [items[round(index*step)] for index in range(TARGET)]
    samples = []
    for item in chosen:
        with Image.open(item['crop']) as handle:
            image = handle.convert('RGB')
        text = read_text(image)
        samples.append({
            'id':item['id'], 'crop':item['crop'], 'crop_size':item['crop_size'],
            'source_image_size':item.get('source_image_size'),
            'ocr_lines':[{'text':line['text'], 'score':line['score'], 'bbox':line['bbox']}
                         for line in text['lines']],
            # 以下为人工核对字段（先留空/占位）
            'verified':False,
            'transcript':' '.join(line['text'] for line in text['lines']),
            'text_ok':None,
            'arrows': [],
            'routes': [],
            'notes':''})
    path = OUTPUTS/'eval_set.json'
    path.write_text(json.dumps({'target':TARGET, 'samples':samples}, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    print(f'评测集候选 {len(samples)} 块 -> {path}')
    for sample in samples:
        print(f"  {sample['id']:>6} {sample['crop_size'][0]:>4}x{sample['crop_size'][1]:<5} "
              f"ocr={sample['transcript'][:64]}")


if __name__ == '__main__':
    main()
