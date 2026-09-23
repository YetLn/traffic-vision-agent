import json
from types import SimpleNamespace
import unittest
from PIL import Image
from traffic_agent.agent import KEEP_MESSAGES, MAX_HISTORY, ask, new_state, run_deepseek
from traffic_agent.tools import ToolContext


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


class FakeMessage:
    def __init__(self, tool=None, arguments='{}', content=None):
        self.content = content
        self.tool_calls = [SimpleNamespace(id='call_1', function=SimpleNamespace(name=tool, arguments=arguments))] if tool else None

    def model_dump(self, **kwargs):
        return {'role':'assistant', 'tool_calls':[{'id':'call_1', 'type':'function',
                'function':vars(self.tool_calls[0].function)}]}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.responses))])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (10,10))
        self.detector = FakeDetector()

    def converse(self, questions, mode='离线规则', client=None):
        state, result = None, None
        for question in questions:
            result = ask(self.detector, self.image, question, mode=mode, state=state, client=client)
            state = result['state']
        return result

    def test_cache_and_followup(self):
        first = ask(self.detector, self.image, '有多少个 ban？')
        second = ask(self.detector, self.image, '把它们框出来', state=first['state'])
        self.assertEqual(self.detector.calls, 1)
        self.assertEqual(second['trace'][0]['arguments'], {'class_name':'ban'})
        self.assertTrue(second['trace'][0]['cached'])

    def test_image_change_and_threshold_invalidate(self):
        first = ask(self.detector, self.image, '有多少个 ban？')
        second = ask(self.detector, Image.new('RGB',(10,10),'white'), '有多少个？', state=first['state'])
        self.assertEqual(len(second['state']['history']), 2)
        self.assertIsNone(second['state']['focus'])
        ask(self.detector, Image.new('RGB',(10,10),'white'), '有多少个？', .5, state=second['state'])
        self.assertEqual(self.detector.calls, 3)

    def test_empty_and_unsupported(self):
        result = ask(FakeDetector(empty=True), self.image, '哪个置信度最高？')
        self.assertIn('没有检出', result['answer'])
        result = ask(self.detector, self.image, '有多少个限速标志？')
        self.assertEqual(result['trace'], [])
        self.assertIn('不能识别限速', result['answer'])
    def test_sessions_isolated_and_bounded(self):
        first = ask(self.detector, self.image, '有多少个 ban？')
        # 同一会话内“它们”解析为上一轮关注的 ban；新会话无从指代，退回全部类别。
        second = ask(self.detector, self.image, '把它们框出来', state=first['state'])
        self.assertEqual(second['trace'][0]['arguments'], {'class_name':'ban'})
        isolated = ask(self.detector, self.image, '把它们框出来')
        self.assertEqual(isolated['trace'][0]['arguments'], {})
        state = second['state']
        for _ in range(10):
            state = ask(self.detector, self.image, '有多少个？', state=state)['state']
        self.assertLessEqual(len(state['history']), MAX_HISTORY)
        self.assertEqual(state['history'][-2], {'role':'user', 'content':'有多少个？'})
        # 折叠把“丢弃旧轮次”换成“把旧轮次计入结构化上下文”。
        self.assertEqual(state['turns'], 12)
        self.assertEqual(state['context']['folded_turns'], 10)
        # 不传 state 即新会话：轮次、历史与折叠上下文都从头开始。
        fresh = ask(self.detector, self.image, '有多少个？')
        self.assertEqual(fresh['state']['turns'], 1)
        self.assertEqual(fresh['state']['history'], [
            {'role':'user','content':'有多少个？'}, {'role':'assistant','content':fresh['answer']}])
        self.assertEqual(fresh['state']['context']['folded_turns'], 0)
        self.assertEqual(fresh['state']['context']['focus'], None)
        # 传入第二个会话的 state 不会读到第一个会话的上下文。
        fresh['state']['history'] = [{'role':'user','content':'会话二'}]
        ask(self.detector, self.image, '有多少个？', state=fresh['state'])
        self.assertEqual(state['history'][-2], {'role':'user', 'content':'有多少个？'})
        self.assertEqual(fresh['state']['history'][-2], {'role':'user', 'content':'有多少个？'})

    def test_tool_validation_before_inference(self):
        ctx = ToolContext(self.detector,self.image,.25,new_state())
        for name,args in [('shell',{}),('count',{'class_name':'speed_limit'}),('detect',{'path':'secret'}),('count',{'class_name':None})]:
            with self.assertRaises(ValueError): ctx.execute(name,args)
        self.assertEqual(self.detector.calls, 0)

    def test_deepseek_protocol_and_result_delivery(self):
        client = FakeClient([FakeMessage('count','{"class_name":"ban"}'),FakeMessage(content='ban 共 1 个。')])
        result = ask(self.detector,self.image,'多少个 ban？',mode='DeepSeek',client=client)
        self.assertEqual(result['answer'],'ban 共 1 个。')
        sent = client.requests[-1]['messages'][-1]
        self.assertEqual(sent['role'],'tool')
        self.assertEqual(json.loads(sent['content'])['count'],1)
        self.assertEqual(client.requests[0]['tool_choice'],'required')

    def test_deepseek_invalid_json_and_loop_bound(self):
        client = FakeClient([FakeMessage('count','{') for _ in range(4)])
        with self.assertRaisesRegex(ValueError,'4 轮'):
            ask(self.detector,self.image,'多少个？',mode='DeepSeek',client=client)
        self.assertEqual(self.detector.calls,0)
        self.assertIn('error',json.loads(client.requests[1]['messages'][-1]['content']))

    def test_missing_inputs(self):
        with self.assertRaises(ValueError): ask(self.detector,None,'检测')
        with self.assertRaises(ValueError): ask(self.detector,self.image,'')

    def test_context_compression_and_bound(self):
        result = self.converse(['有多少个 ban？'] * 10)
        state = result['state']
        self.assertLessEqual(len(state['history']), MAX_HISTORY)
        self.assertEqual(state['turns'], 10)
        self.assertTrue(all(row['role'] != 'system' for row in state['history']))
        context = state['context']
        self.assertEqual(context['folded_turns'], 8)
        self.assertEqual(context['first_detection']['ban'], 1)
        self.assertEqual(context['fold_keys'], [state['cache_key']])
        self.assertEqual(len(context['detected_images']), 1)
        self.assertTrue(context['detected_images'][0].startswith('内存图片-'))
        self.assertEqual(context['recent_facts'][-1]['total'], 1)
        self.assertNotIn('PIL', str(result['context']))
        self.assertIn('已折叠轮次', result['context'])
        self.assertEqual(result['context']['当前关注类别'], 'ban')
        self.assertEqual(self.detector.calls, 1)

    def test_context_answers_are_isolated_from_image_facts(self):
        first = self.converse(['图里有什么交通标志？'])
        early = ask(self.detector, self.image, '之前的对话摘要是什么？', state=first['state'])
        self.assertIn('还没有发生上下文折叠', early['answer'])
        folded = self.converse(['有多少个 ban？'] * 4)['state']
        followup = ask(self.detector, self.image, '之前的对话摘要是什么？', state=folded)
        self.assertTrue(followup['answer'].startswith('已折叠的会话上下文'), followup['answer'])
        self.assertIn('ban', followup['answer'])
        image_question = ask(self.detector, self.image, '这张图片里有什么？', state=followup['state'])
        self.assertIn('检测到', image_question['answer'])

    def test_deepseek_receives_folded_summary(self):
        previous = self.converse(['有多少个 ban？'] * 4)
        state = previous['state']
        client = FakeClient([FakeMessage(content='本轮没有调用工具。')])
        result = ask(self.detector,self.image,'之前的对话摘要是什么？',mode='DeepSeek',state=state,client=client)
        self.assertEqual(result['answer'], '本轮没有调用工具。')
        request = [m for m in client.requests[-1]['messages'] if m['role']=='system' and '已折叠' in m['content']]
        self.assertTrue(request)
        self.assertIn('ban', request[-1]['content'])
        self.assertEqual(client.requests[-1]['messages'][-1], {'role':'user','content':'之前的对话摘要是什么？'})
        self.assertEqual(self.detector.calls, 1)

    def test_ordinal_target_does_not_reuse_highest_or_route_to_other_tools(self):
        class TwoTargetDetector(FakeDetector):
            def detect(self, image, conf):
                self.calls += 1
                return {'detections': [
                    dict(class_id=1, class_name='ban', confidence=.95, bbox=[1,1,3,3]),
                    dict(class_id=0, class_name='wran', confidence=.65, bbox=[5,5,9,9]),
                ], 'total': 2, 'counts': {'ban': 1, 'wran': 1}}

        detector = TwoTargetDetector()
        first = ask(detector, self.image, '哪个置信度最高？')
        self.assertEqual(first['trace'][0]['result']['detection']['confidence'], .95)
        state = first['state']
        for question in ('第二个的置信度是多少？', '第2个标志是什么？',
                         '把第二个目标框出来', '第二块牌上写了什么？',
                         '第 2 个标志是什么意思？', '第三个目标呢？'):
            with self.subTest(question=question):
                result = ask(detector, self.image, question, state=state)
                self.assertIn('尚不支持', result['answer'])
                self.assertIn('裁剪', result['answer'])
                self.assertNotIn('0.9500', result['answer'])
                self.assertEqual(result['trace'], [])
                self.assertIsNone(result['image'])
                self.assertEqual(detector.calls, 1)

    def test_explicit_highest_confidence_followup_still_uses_cached_tool(self):
        first = ask(self.detector,self.image,'有多少个 ban？')
        result = ask(self.detector,self.image,'哪个置信度最高？',state=first['state'])
        self.assertIn('最高置信度', result['answer'])
        self.assertEqual(result['trace'][0]['tool'], 'highest_confidence')
        self.assertTrue(result['trace'][0]['cached'])
        self.assertEqual(self.detector.calls, 1)


if __name__ == '__main__':
    unittest.main()
