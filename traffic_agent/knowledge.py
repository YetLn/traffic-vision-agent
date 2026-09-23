"""交通标志语义知识库：本地 YAML + 关键词检索，不依赖向量库或外部 API。

设计纪律（对应自身能力的诚实边界）：
- 每个类别的语义字段全部来自 `knowledge/signs.yaml`，未填写即视为不可用；
- 证据不足时返回 available=False，Agent 只能回答“知识库无依据”，不得凭外观猜测；
- 这是关键词/同义检索（retrieval），不是 FAISS 向量检索，也不做 embedding。

用法：
    from traffic_agent.knowledge import inventory, lookup, answer
    lookup('ban')          # -> {'available': False, 'reason': '知识库缺少...'}
    answer('ban 是什么意思？')  # -> 面向用户的诚实回答
"""

from functools import lru_cache
from pathlib import Path
import re

import yaml

from .config import ROOT

KNOWLEDGE_FILE = ROOT / 'knowledge' / 'signs.yaml'
WRAN, BAN, POINT_L, POINT_S = 'wran', 'ban', 'point-l', 'point-s'
# 必须与 checkpoint 的类别名一致；语义待确认，此处只做检索用的别名。
ALIASES = {WRAN: [], BAN: [], POINT_L: [], POINT_S: []}
FIELDS = ('meaning', 'appearance', 'scene', 'confusable')
PENDING = '待确认'
UNKNOWN = ('当前知识库没有该标志的权威语义依据，不能凭外观或常识推测。'
           '请补充类别含义与依据来源（如国标条款号或项目标注手册）后再查询。')
NO_QUESTION = '请说明要查询的类别，例如“ban 是什么意思？”。'


@lru_cache(maxsize=1)
def entries():
    """读取知识库并做结构校验；文件缺失或类别不符会直接报错而不是静默降级。"""
    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f'找不到知识库文件：{KNOWLEDGE_FILE}')
    data = yaml.safe_load(KNOWLEDGE_FILE.read_text(encoding='utf-8')) or {}
    rows = data.get('classes') or []
    found = [str(row.get('class_name', '')).strip() for row in rows]
    if sorted(found) != sorted(ALIASES):
        raise ValueError(f'知识库类别必须与模型类别一致，当前为 {found}')
    return tuple({'class_name': name, 'label': str(row.get('label', name)),
                  'aliases': [str(a) for a in (row.get('aliases') or [])],
                  'reference': str(row.get('reference') or '').strip(),
                  'meaning': str(row.get('meaning') or '').strip(),
                  'appearance': str(row.get('appearance') or '').strip(),
                  'scene': str(row.get('scene') or '').strip(),
                  'confusable': str(row.get('confusable') or '').strip()}
                 for name, row in zip(found, rows))


@lru_cache(maxsize=1)
def _index():
    return {row['class_name']: row for row in entries()}


def inventory():
    """给界面/报告用的知识库状态：哪些类别已确认、哪些还缺依据。"""
    rows = []
    for row in entries():
        available = is_available(row)
        rows.append({'class_name':row['class_name'], 'label':row['label'],
                     'available':available, 'reference':row['reference'],
                     'fields':[name for name in FIELDS if row[name]]})
    return {'available_count':sum(1 for r in rows if r['available']), 'total':len(rows),
            'classes':rows}


def is_available(row):
    """有含义且有依据才算可用；只有含义没依据同样视为待确认。"""
    return bool(row['meaning']) and bool(row['reference']) and row['reference'] != PENDING


def _dedupe(values):
    seen, out = set(), []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def candidates(question):
    """对知识库做关键词/别名检索，返回“行 + 命中词”。"""
    text = (question or '').lower()
    hits = []
    for row in entries():
        matched = []
        for token in _dedupe([row['class_name'], row['label'], *row['aliases']]):
            if token and token.lower() in text:
                matched.append(token)
        low = text
        score = sum(2 if token.lower() in (row['class_name'], row['label']) else 1
                    for token in matched)
        if any(word in low for word in ('指示', '指路', '导向', '方向')) and row['class_name'].startswith('point'):
            score += 1
        if score:
            hits.append((score, row, matched))
    return sorted(hits, key=lambda item: -item[0])


def lookup(class_name, question=None):
    """按类别（可结合问题做同义匹配）取语义；证据不足一律返回 available=False。"""
    if not class_name:
        return {'available':False, 'class_name':None, 'reason':NO_QUESTION}
    row = _index().get(str(class_name).strip())
    if row is None:
        return {'available':False, 'class_name':class_name,
                'reason':f'模型不支持类别 {class_name}；不能推断限速值或具体标志含义。'}
    result = {'available':False, 'class_name':row['class_name'], 'label':row['label'],
              'reference':row['reference']}
    if not is_available(row):
        result['reason'] = UNKNOWN
        return result
    result.update({'available':True, 'meaning':row['meaning'], 'appearance':row['appearance'],
                   'scene':row['scene'], 'confusable':row['confusable'], 'aliases':row['aliases']})
    return result


def answer(question, class_name=None):
    """离线规则模式下的语义回答：只用知识库内容，缺依据就明说。"""
    matched = candidates(question or '')
    if class_name is None and matched:
        class_name = matched[0][1]['class_name']
    if class_name is None:
        hint = '、'.join(row['class_name'] for row in entries())
        return f'{NO_QUESTION}可选类别：{hint}。'
    result = lookup(class_name, question)
    if not result['available']:
        return result.get('reason', UNKNOWN)
    parts = [f"{result['class_name']}（{result['label']}）：{result['meaning']}"]
    if result['appearance']:
        parts.append(f"外观：{result['appearance']}")
    if result['scene']:
        parts.append(f"典型场景：{result['scene']}")
    if result['confusable']:
        parts.append(f"易混淆：{result['confusable']}")
    parts.append(f"依据：{result['reference']}")
    parts.append('以上仅为标志语义，不含行车安全判断，也不代表图中一定存在该类标志。')
    return ' '.join(parts)


def reference_text(class_name):
    """回传给模型的知识库文本；不可用时返回空串。"""
    result = lookup(class_name)
    if not result['available']:
        return ''
    return '；'.join(f'{key}={result[key]}' for key in FIELDS if result.get(key))


def strip_placeholders(text):
    """把未确认的占位符去掉，避免把“待确认”当成依据写进报告。"""
    return re.sub(r'\s*待确认\s*', '', text or '').strip()
