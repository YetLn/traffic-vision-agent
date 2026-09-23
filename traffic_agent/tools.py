import uuid
from . import knowledge
from .config import OUTPUTS
from .detector import image_key
from .sign_text import inference_prompt, read_signs, summarize

DESCRIPTIONS = {
    'detect': '检测当前图片，返回类别、数量、置信度与边框。',
    'count': '统计当前图片的全部目标，或指定原始类别的数量。',
    'highest_confidence': '返回当前图片置信度最高的检测目标。',
    'visualize': '绘制当前图片全部或指定类别的检测框。',
    'lookup_sign': '查询指定类别交通标志的含义。只返回本地知识库中已有依据的内容；没有依据时返回 available=false。',
    'read_sign_text': '对图中 point-l（指路标志）按原图分辨率裁剪并识别牌面文字，返回逐段 OCR 原文与置信度。只返回识别到的文字，不做含义推断。',
}


def schemas(names):
    names = list(names)
    tools = []
    for name, description in DESCRIPTIONS.items():
        properties = {}
        if name in ('count', 'visualize'):
            properties['class_name'] = {'type': 'string', 'enum': names,
                                         'description': '省略表示全部类别。'}
        if name == 'lookup_sign':
            properties['class_name'] = {'type': 'string', 'enum': names,
                                        'description': '要查询语义的标志类别。'}
        tools.append({'type': 'function', 'function': {'name': name,
                      'description': description, 'parameters': {
                          'type': 'object', 'properties': properties,
                          'additionalProperties': False}}})
    return tools


class ToolContext:
    def __init__(self, detector, image, threshold, state):
        self.detector, self.image = detector, image
        self.threshold, self.state = threshold, state
        self.trace = []
        self.visualization = None

    def execute(self, name, arguments):
        if name not in DESCRIPTIONS:
            raise ValueError(f'未知工具：{name}')
        allowed = {'class_name'} if name in ('count', 'visualize', 'lookup_sign') else set()
        if not isinstance(arguments, dict) or set(arguments) - allowed:
            raise ValueError('工具参数不符合定义。')
        category = arguments.get('class_name')
        if 'class_name' in arguments and category not in self.detector.names.values():
            raise ValueError('模型不支持该类别；不能推断限速值或具体标志含义。')
        if name == 'lookup_sign':
            if not category:
                raise ValueError('查询标志语义需要指定类别。')
            output = knowledge.lookup(category)
            self.trace.append({'tool': name, 'arguments': arguments, 'cached': False,
                               'result': output})
            return output
        if name == 'read_sign_text':
            signs = read_signs(self.image, self.detector, self.threshold)
            output = {'sign_count':len(signs), 'signs':summarize(signs),
                      'prompt':inference_prompt(signs) if signs else '',
                      'empty_reason':None if signs else
                      '本图未检出 point-l（指路标志），无法提取牌面文字。'}
            self.trace.append({'tool': name, 'arguments': arguments, 'cached': False,
                               'result': output})
            return output
        key = image_key(self.image, self.threshold)
        cached = self.state.get('cache_key') == key
        if not cached:
            self.state['result'] = self.detector.detect(self.image, self.threshold)
            self.state['cache_key'] = key
        data = self.state['result']
        if name == 'detect':
            output = data
        elif name == 'count':
            output = {'class_name': category or 'all',
                      'count': data['counts'].get(category, 0) if category else data['total'],
                      'counts': data['counts']}
        elif name == 'highest_confidence':
            output = {'detection': max(data['detections'], key=lambda x:x['confidence'], default=None)}
        else:
            boxes = [d for d in data['detections'] if not category or d['class_name'] == category]
            self.visualization = self.detector.visualize(
                self.image, boxes, OUTPUTS / f'{uuid.uuid4().hex}.jpg')
            output = {'status': 'generated', 'boxes': len(boxes), 'class_name': category or 'all'}
        self.trace.append({'tool': name, 'arguments': arguments, 'cached': cached, 'result': output})
        return output

