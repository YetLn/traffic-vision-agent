"""知识库框架测试：结构校验、检索、缺依据拒答，以及 Agent 侧的工具接入。

这些测试用打桩替换知识库内容，因此不依赖 signs.yaml 是否已填写语义，
框架本身的正确性始终可验证。
"""

import unittest
from unittest import mock
from PIL import Image

from traffic_agent import agent, knowledge, sign_text
from traffic_agent.tools import ToolContext, schemas


FILLED = (
    {'class_name':knowledge.WRAN, 'label':'wran', 'aliases':['警告'], 'reference':'GB/T 测试 4.1',
     'meaning':'警告标志', 'appearance':'三角形', 'scene':'危险路段', 'confusable':''},
    {'class_name':knowledge.BAN, 'label':'ban', 'aliases':['禁止'], 'reference':'GB/T 测试 4.2',
     'meaning':'禁止标志', 'appearance':'圆形', 'scene':'禁行路段', 'confusable':'与指示标志区分'},
    {'class_name':knowledge.POINT_L, 'label':'point-l', 'aliases':[], 'reference':knowledge.PENDING,
     'meaning':'指示/指路标志（大尺寸，含依据的类别）', 'appearance':'', 'scene':'', 'confusable':''},
    {'class_name':knowledge.POINT_S, 'label':'point-s', 'aliases':[], 'reference':knowledge.PENDING,
     'meaning':'', 'appearance':'', 'scene':'', 'confusable':''},
)


PENDING_REASON = '打桩：没有该标志的权威语义依据'
HINT = '打桩：请说明要查询的类别'
UNSUPPORTED = '打桩：不支持该类别'


class knowledge_patch:
    """替换索引与查询函数，保证 tools/agent/直接调用三条路径使用同一份打桩数据。"""

    def __init__(self, rows_):
        self.rows = {row['class_name']:row for row in rows_}

    def _category(self, question, class_name):
        if class_name:
            return class_name
        text = (question or '').lower()
        return next((name for name in self.rows if name in text), None)

    def _reason(self, row):
        return PENDING_REASON

    def __enter__(self):
        def is_available(row):
            return bool(row['meaning']) and bool(row['reference']) and row['reference'] != knowledge.PENDING

        def lookup(class_name=None, question=None):
            class_name = self._category(question, class_name)
            if not class_name:
                return {'available':False, 'class_name':None, 'reason':HINT}
            row = self.rows.get(class_name)
            if row is None:
                return {'available':False, 'class_name':class_name, 'reason':UNSUPPORTED}
            if not is_available(row):
                return {'available':False, 'class_name':class_name, 'reason':self._reason(row)}
            return {'available':True, 'class_name':row['class_name'], 'label':row['label'],
                    'meaning':row['meaning'], 'appearance':row['appearance'],
                    'scene':row['scene'], 'confusable':row['confusable'],
                    'reference':row['reference']}

        def answer(question='', class_name=None):
            result = lookup(class_name, question)
            if not result['available']:
                return result['reason']
            return f"{result['class_name']}：{result['meaning']} 依据：{result['reference']}"

        def find_category(question, names, state=None):
            found = next((name for name in self.rows if name in (question or '').lower()), None)
            return found if found in names else None

        self.patchers = [mock.patch.object(knowledge, name, new=func) for name, func in (
            ('entries', lambda: tuple(self.rows.values())), ('_index', lambda: self.rows),
            ('is_available', is_available), ('lookup', lookup), ('answer', answer))] + \
            [mock.patch.object(agent, 'find_category', new=find_category)]
        for patcher in self.patchers:
            patcher.start()
        return self

    def __exit__(self, *exc_info):
        for patcher in reversed(self.patchers):
            patcher.stop()


class KnowledgeTests(unittest.TestCase):
    def test_file_structure_matches_model_classes(self):
        entries = knowledge.entries()
        self.assertEqual(sorted(row['class_name'] for row in entries), sorted(knowledge.ALIASES))

    def test_lookup_rejects_unknown_class(self):
        result = knowledge.lookup('speed_limit')
        self.assertFalse(result['available'])
        self.assertIn('不支持类别', result['reason'])

    def test_lookup_requires_reference(self):
        with knowledge_patch(FILLED):
            result = knowledge.lookup(knowledge.POINT_L)
        self.assertFalse(result['available'])
        self.assertIn(PENDING_REASON, result['reason'])

    def test_lookup_and_answer_use_filled_entry(self):
        with knowledge_patch(FILLED):
            hit = knowledge.lookup(knowledge.BAN, '禁止标志是什么意思')
            self.assertTrue(hit['available'])
            self.assertEqual(hit['meaning'], '禁止标志')
            text = knowledge.answer('ban 是什么意思？')
            self.assertIn('禁止标志', text)
            self.assertIn('依据：GB/T 测试 4.2', text)
            pending = knowledge.answer('point-l 是什么意思？')
            self.assertIn('没有该标志的权威语义依据', pending)

    def test_inventory_reports_pending(self):
        with knowledge_patch(FILLED):
            data = knowledge.inventory()
        self.assertEqual(data['available_count'], 2)
        self.assertEqual([r['class_name'] for r in data['classes'] if not r['available']],
                         [knowledge.POINT_L, knowledge.POINT_S])

    def test_offline_semantic_question_never_invents_meaning(self):
        image, detector = Image.new('RGB', (10, 10)), FakeDetector()
        with knowledge_patch(FILLED):
            no_category = agent.ask(detector, image, '这个标志是什么意思？')
            self.assertIn(HINT, no_category['answer'])
            for question in ['point-l 是什么意思？', 'point-s 表示什么？']:
                result = agent.ask(detector, image, question)
                self.assertIn(PENDING_REASON, result['answer'], question)
                self.assertEqual(result['trace'], [], question)
            unsupported = agent.ask(detector, image, '有多少个限速标志？')
            self.assertIn('不能识别限速', unsupported['answer'])
            self.assertEqual(unsupported['trace'], [])
        self.assertEqual(detector.calls, 0)

    def test_offline_answer_uses_knowledge_when_available(self):
        image, detector = Image.new('RGB', (10, 10)), FakeDetector()
        with knowledge_patch(FILLED):
            result = agent.ask(detector, image, 'ban 是什么意思？')
        self.assertEqual(result['trace'][0]['tool'], 'lookup_sign')
        self.assertIn('依据：GB/T 测试 4.2', result['answer'])
        self.assertEqual(detector.calls, 0)

    def test_knowledge_lookup_uses_detected_class(self):
        image, detector = Image.new('RGB', (10, 10)), FakeDetector()
        first = agent.ask(detector, image, '图里有什么交通标志？')
        called = {}

        def fake_lookup(class_name, question=None):
            called['class_name'] = class_name
            return {'available':False, 'class_name':class_name, 'reason':knowledge.UNKNOWN}

        with mock.patch.object(knowledge, 'lookup', side_effect=fake_lookup):
            result = agent.ask(detector, image, '这个标志表示什么？', state=first['state'])
        self.assertEqual(called['class_name'], 'ban')
        self.assertIn('没有该标志的权威语义依据', result['answer'])


class FakeDetector:
    names = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s'}

    def __init__(self, empty=False):
        self.calls = 0
        self.empty = empty

    def detect(self, image, conf):
        self.calls += 1
        ds = [] if self.empty else [dict(class_id=1, class_name='ban', confidence=.8, bbox=[1,2,3,4])]
        return {'detections':ds, 'total':len(ds), 'counts':{'ban':1} if ds else {}}

    def visualize(self, image, detections, path):
        self.drawn = detections
        return str(path)


class SignTextDetector(FakeDetector):
    """报告的检测框覆盖整张图，用于驱动 read_sign_text 的裁剪与 OCR。"""

    def detect(self, image, conf):
        self.calls += 1
        width, height = image.size
        return {'detections':[dict(class_id=2, class_name='point-l', confidence=.9,
                                   bbox=[0, 0, width, height])],
                'total':1, 'counts':{'point-l':1}}


class SignTextTests(unittest.TestCase):
    def fake_ocr(self, result):
        return mock.patch.object(sign_text, 'read_text', return_value=result)

    def test_splits_prompts_from_place_names(self):
        lines = [{'text':'前方施工 减速慢行', 'score':0.9, 'bbox':[0,0,10,10], 'height':30},
                 {'text':'平安路', 'score':0.9, 'bbox':[0,0,10,10], 'height':90},
                 {'text':'Pingan Rd', 'score':0.9, 'bbox':[0,0,10,10], 'height':30},
                 {'text':'迎宾务', 'score':0.3, 'bbox':[0,0,10,10], 'height':30}]
        buckets = sign_text.split_lines(lines)
        self.assertIn('前方施工 减速慢行', [line['text'] for line in buckets['prompts']])
        self.assertIn('平安路', [line['text'] for line in buckets['destinations']])
        self.assertIn('Pingan Rd', [line['text'] for line in buckets['supporting']])
        self.assertEqual(buckets['low_confidence'][0]['text'], '迎宾务')

    def test_small_crop_is_not_readable(self):
        detector = SignTextDetector()
        with Image.new('RGB', (75, 118)) as tiny:
            signs = sign_text.read_signs(tiny, detector)
        self.assertEqual(len(signs), 1)
        self.assertFalse(signs[0]['readable'])
        self.assertIn('可读门槛', signs[0]['readable_reason'])
        self.assertEqual(signs[0]['texts'], [])

    def test_readable_crop_returns_text_and_prompt(self):
        detector = SignTextDetector()
        ocr = {'lines':[{'text':'平安路', 'score':0.9, 'bbox':[0,0,10,10], 'height':90}],
               'elapsed_ms':1.0}
        with Image.new('RGB', (400, 300)) as image:
            with mock.patch.object(sign_text, 'read_text', return_value=ocr):
                signs = sign_text.read_signs(image, detector)
        self.assertTrue(signs[0]['readable'])
        self.assertEqual(signs[0]['texts'], ['平安路'])
        prompt = sign_text.inference_prompt(signs)
        self.assertIn('推测必须标明是推测', prompt)
        self.assertIn('平安路', prompt)

    def test_unreadable_sign_is_declared_in_prompt(self):
        detector = SignTextDetector()
        with Image.new('RGB', (80, 80)) as tiny:
            signs = sign_text.read_signs(tiny, detector)
        prompt = sign_text.inference_prompt(signs)
        self.assertIn('不可读', prompt)
        self.assertIn('不得根据检测类别或外观猜测内容', prompt)

    def test_offline_answer_reports_text_without_inference(self):
        detector = SignTextDetector()
        ocr = {'lines':[{'text':'翠园路', 'score':0.9, 'bbox':[0,0,10,10], 'height':90}],
               'elapsed_ms':1.0}
        with Image.new('RGB', (400, 300)) as image:
            with mock.patch.object(sign_text, 'read_text', return_value=ocr):
                result = agent.ask(detector, image, '牌子上写的什么文字？')
        self.assertEqual(result['trace'][0]['tool'], 'read_sign_text')
        self.assertIn('翠园路', result['answer'])
        self.assertIn('未做含义推断', result['answer'])

    def test_prompt_and_place_classification(self):
        cases = [('请按导向车道行驶', 'prompt'), ('各行其道', 'prompt'), ('禁止通行', 'prompt'),
                 ('减速慢行', 'prompt'), ('大客停车场', 'info'), ('游客中心', 'info'),
                 ('平安路', 'place'), ('G5513', 'other'), ('秦', 'other'),
                 # 留出集暴露的误判：单位名与赛事标识不得判为提示语
                 ('车辆管理分所', 'info'), ('公路自行车赛场', 'info'),
                 ('Pingan Rd', 'other'), ('美篇', 'other')]
        for text, expected in cases:
            kind, _ = sign_text.classify_text(text)
            self.assertEqual(kind, expected, f'{text} 期望 {expected} 实际 {kind}')

    def test_generic_words_alone_are_not_prompts(self):
        # 单独出现“车辆/行车/车道”不足以判定为行车要求
        for text in ('车辆管理分所', '公路自行车赛场', '停车库'):
            kind, _ = sign_text.classify_text(text)
            self.assertNotEqual(kind, 'prompt', text)
        self.assertTrue(sign_text.is_prompt_text('右转车辆 借用公交车道'))

    def test_split_lines_extracts_prompt_bucket(self):
        lines = [{'text':'请按导向车道行驶', 'score':0.95, 'bbox':[0,0,10,10], 'height':40},
                 {'text':'地下停车场', 'score':0.95, 'bbox':[0,0,10,10], 'height':40},
                 {'text':'嘉州大道', 'score':0.95, 'bbox':[0,0,10,10], 'height':90}]
        buckets = sign_text.split_lines(lines)
        self.assertEqual([line['text'] for line in buckets['prompts']], ['请按导向车道行驶'])
        self.assertEqual([line['text'] for line in buckets['destinations']], ['嘉州大道'])

    def test_tool_context_exposes_read_sign_text(self):
        detector = SignTextDetector()
        ocr = {'lines':[{'text':'嘉祥路', 'score':0.9, 'bbox':[0,0,10,10], 'height':90}],
               'elapsed_ms':1.0}
        with Image.new('RGB', (400, 300)) as image:
            context = ToolContext(detector, image, .25, agent.new_state())
            with mock.patch.object(sign_text, 'read_text', return_value=ocr):
                output = context.execute('read_sign_text', {})
        self.assertEqual(output['sign_count'], 1)
        self.assertIn('嘉祥路', output['prompt'])
        names = {tool['function']['name'] for tool in schemas(detector.names.values())}
        self.assertIn('read_sign_text', names)


if __name__ == '__main__':
    unittest.main()
