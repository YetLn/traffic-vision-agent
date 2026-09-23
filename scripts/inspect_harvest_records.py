"""检查抽样报告中指定 id 的原始记录（裁剪路径、文字、分类结果）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_agent.config import OUTPUTS

targets = set(sys.argv[1:]) or {'08325', '08525', '08086', '08274', '08327', '08504'}
data = json.loads((OUTPUTS/'prompt_sign_candidates.json').read_text(encoding='utf-8'))
print(f"报告记录数 {len(data['records'])}，提示类 {data['prompt_records']}")
for record in data['records']:
    if record['id'] not in targets:
        continue
    path = Path(record['crop'])
    print(f"--- {record['id']} {record['crop_size']} 存在={path.is_file()}")
    print(f"    path={record['crop']}")
    print(f"    texts={' / '.join(record.get('texts', []))}")
    print(f"    prompt_like={record.get('prompt_like')} type={record.get('type')}")
