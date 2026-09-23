"""列出抽样结果的 OCR 文字，用于人工核对提示/规则类文本覆盖情况。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import OUTPUTS

data = json.loads((OUTPUTS/'prompt_sign_candidates.json').read_text(encoding='utf-8'))
records = data['records']
print(f"裁剪总数 {len(records)}，抽样标注数 {data['sampled_ids']}")
print(f"主色分布 {data['dominant_histogram']}，非蓝底命中 {data['kept']}")
print('=== 提示/规则类命中全文 ===')
for record in records:
    if record.get('prompt_like'):
        print(f"  {record['id']} | " + ' / '.join(record['texts']))
others = [record for record in records if record.get('type') == '其他/待判']
print(f"=== 其他/待判（{len(others)} 条）前 40 条 ===\n")
for record in others[:40]:
    print(f"  {record['id']} | " + ' / '.join(record.get('texts', []))[:110])
