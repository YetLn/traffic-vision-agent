"""按颜色筛选非蓝底文字牌（规则/施工/警示类提示牌），并导出清单。

背景：point-l 类别定义是“蓝绿棕底、含文字的指路标志”，因此施工/规则类提示牌大多
是黄底或红/白底。要对它们做文字覆盖，先按牌面主色把它们从蓝底指路牌里分出来。

用法：
    .venv\\Scripts\\python.exe scripts\\collect_prompt_signs.py --sample 260
输出：
    outputs/prompt_sign_candidates.json  候选清单（含颜色统计）
    outputs/prompt_sign_crops/           命中的牌面裁剪
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from traffic_agent.config import OUTPUTS

# HSV 主色区间（OpenCV: H 0-179, S/V 0-255）
COLOR_RANGES = {
    'blue':  [((100, 60, 40), (130, 255, 255))],
    'green': [((40, 60, 40), (95, 255, 255))],
    'yellow':[((15, 60, 90), (38, 255, 255))],
    'red':   [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (179, 255, 255))],
    'white': [((0, 0, 170), (179, 60, 255))],
}


def color_shares(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0]*hsv.shape[1]
    shares = {}
    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        shares[name] = round(float((mask > 0).sum())/total, 3)
    # 蓝绿的“指路牌底色”
    shares['directional_background'] = round(shares['blue'] + shares['green'], 3)
    shares['rule_or_warning_background'] = round(shares['yellow'] + shares['red'] + shares['white'], 3)
    return shares


def is_non_directional(shares, min_yellow_red=0.05, min_white=0.30, max_directional=0.25):
    """判断是否为“非蓝色指路牌”：黄/红底，或浅色底且蓝绿占比低。

    注意不能只看白色占比——蓝底指路牌的文字与边框本身就是白色，会误判为提示牌。
    实测黄/红占比只有几个百分点时多为反光与边框噪声，故阈值取 0.15。
    """
    if shares['yellow'] + shares['red'] >= min_yellow_red:
        return True, '黄/红底'
    if shares['white'] >= min_white and shares['directional_background'] <= max_directional:
        return True, '浅色底且蓝绿占比低'
    return False, None


def main():
    parser = argparse.ArgumentParser(description='按颜色筛选非蓝底文字牌')
    parser.add_argument('--sample', type=int, default=60, help='均匀抽样多少个标注候选')
    parser.add_argument('--crops-per-id', type=int, default=2, help='每个标注最多取几张裁剪')
    parser.add_argument('--min-rule', type=float, default=0.15,
                        help='黄+红占比阈值')
    parser.add_argument('--ocr', action='store_true', help='对命中项跑 OCR 并按文字分类')
    parser.add_argument('--measure-yield', action='store_true',
                        help='对所有抽样项跑 OCR，统计“提示/规则类文字”的产出率（耗时长）')
    parser.add_argument('--start', type=int, default=0,
                        help='从均匀抽样的第几个位置开始，用于留出集（避免与调词样本重叠）')
    parser.add_argument('--tag', default='prompt_sign_candidates',
                        help='报告文件名（不含扩展名），便于区分调词集与留出集')
    args = parser.parse_args()

    from traffic_agent.sign_text import classify_text, is_place_name
    all_crops = json.loads((OUTPUTS/'ocr_samples.json').read_text(encoding='utf-8'))['crops']
    grouped = {}
    for item in all_crops:
        grouped.setdefault(item['id'], []).append(item)
    for group in grouped.values():
        group.sort(key=lambda item: -item['crop_size'][0])
    ids = sorted(grouped)
    step = max(1, len(ids)//args.sample)
    pool = ids[::step]
    selected_ids = pool[args.start:args.start + args.sample]
    if args.measure_yield:
        items = [item for name in selected_ids for item in grouped[name][:args.crops_per_id]]
    else:
        items = [grouped[name][0] for name in selected_ids]

    out_dir = OUTPUTS/'prompt_sign_crops'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    records = []
    for item in items:
        with Image.open(item['crop']) as handle:
            image = handle.convert('RGB')
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        shares = color_shares(bgr)
        dominant = max(('blue', 'green', 'yellow', 'red'), key=lambda key: shares[key])
        summary[dominant] += 1
        keep, reason = is_non_directional(shares, args.min_rule)
        record = {'id':item['id'], 'crop':item['crop'], 'crop_size':item['crop_size'],
                  'shares':shares, 'dominant':dominant, 'kept':keep, 'reason':reason}
        if keep:
            target = out_dir/Path(item['crop']).name
            image.save(target, quality=95)
            record['kept_crop'] = str(target)
        if args.ocr or args.measure_yield:
            from traffic_agent.ocr import read_text
            ocr = read_text(image)
            texts = [line['text'] for line in ocr['lines']]
            record['texts'] = texts
            record['prompt_like'] = [text for text in texts
                                     if classify_text(text)[0] == 'prompt']
            record['place_like'] = [text for text in texts if is_place_name(text)]
            record['type'] = ('提示/规则类' if record['prompt_like']
                              else ('地名指路类' if record['place_like'] else '其他/待判'))
        records.append(record)

    kept = [record for record in records if record['kept']]
    prompt_records = [record for record in records if record.get('prompt_like')]
    types = Counter(record.get('type') for record in records) if (args.ocr or args.measure_yield) else {}
    report = {'sampled_ids':len(selected_ids), 'start_offset':args.start,
              'crops_processed':len(records),
              'dominant_histogram':dict(summary), 'kept':len(kept),
              'min_rule_share':args.min_rule, 'type_histogram':dict(types),
              'prompt_records':len(prompt_records), 'records':records}
    path = OUTPUTS/f'{args.tag}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'抽样 {len(selected_ids)} 个标注 / 处理 {len(records)} 张裁剪')
    print(f'主色分布 {dict(summary)}；非蓝底命中 {len(kept)} 张')
    if types:
        print(f'按文字分类 {dict(types)}；提示/规则类文字命中 {len(prompt_records)} 张')
        for record in prompt_records:
            print(f"  [提示类] {record['id']:>6} {' / '.join(record['texts'])[:90]}")
    print(f'报告：{path}')


if __name__ == '__main__':
    main()

