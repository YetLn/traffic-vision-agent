import types
import unittest

from traffic_agent import rag


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.kwargs = None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class RagBoundaryTests(unittest.TestCase):
    def test_corpus_contains_only_public_allowlisted_sources(self):
        sources = {row['source'] for row in rag.corpus()}
        self.assertEqual(sources, {*rag.SOURCES, 'knowledge/signs.yaml'})
        self.assertGreater(len(rag.corpus()), 20)
        self.assertTrue(all(row['line'] > 0 and row['text'] for row in rag.corpus()))

    def test_visual_fact_never_reaches_retriever_or_llm(self):
        client = FakeClient('图里有牌。')
        result = rag.answer('这张图片里的牌面写了什么？', 'DeepSeek 引用回答', client)
        self.assertFalse(result['available'])
        self.assertEqual(result['answer'], rag.IMAGE_BOUNDARY)
        self.assertIsNone(client.kwargs)

    def test_generation_requires_existing_citation(self):
        question = '方向解析为什么还没有通过验收？'
        citation = rag.search(question)['hits'][0]['id']
        valid = FakeClient(f'目前只有开发性质的结果 [{citation}]。')
        result = rag.answer(question, 'DeepSeek 引用回答', valid)
        self.assertEqual(result['used_citations'], [citation])
        self.assertNotIn('.runtime', str(valid.kwargs['messages']))
        invalid = FakeClient('方向解析已经通过正式验收。')
        refused = rag.answer(question, 'DeepSeek 引用回答', invalid)
        self.assertEqual(refused['answer'], rag.NO_EVIDENCE)
        wrong = FakeClient('答案见 [Kffffffff]。')
        self.assertEqual(rag.answer(question, 'DeepSeek 引用回答', wrong)['answer'],
                         rag.NO_EVIDENCE)


if __name__ == '__main__':
    unittest.main()
