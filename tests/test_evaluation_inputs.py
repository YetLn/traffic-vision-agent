"""评测输入必须完整；无效样本不得触发推理或覆盖已有成绩。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from scripts import evaluate_sign_parser as evaluation


def case(case_id, crop):
    return {'id': case_id, 'crop': crop, 'difficulty': '测试',
            'routes': [{'destination': '测试路', 'direction': 'up'}],
            'ambiguous': [], 'notes': ''}


class EvaluationInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = self.root / 'samples'
        self.samples.mkdir()
        Image.new('RGB', (20, 20), 'blue').save(self.samples / 'valid.png')
        self.output = self.root / 'report.json'
        self.old_report = b'{"samples": 20, "previous": true}\n'
        self.output.write_bytes(self.old_report)

    def run_main(self, cases):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(evaluation, 'EVAL_SET', cases), \
             patch.object(evaluation, 'read_text') as ocr, \
             patch.object(evaluation, 'parse_sign') as parse, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ocr.return_value = {'lines': [{'text': '测试路'}]}
            parse.return_value = {'routes': [{'destination': '测试路', 'direction': 'up'}]}
            result = evaluation.main(['--samples', str(self.samples), '--output', str(self.output)])
        return result, stderr.getvalue(), ocr, parse

    def assert_rejected(self, cases, message):
        mtime = self.output.stat().st_mtime_ns
        status, stderr, ocr, parse = self.run_main(cases)
        self.assertEqual(status, 2)
        self.assertIn(message, stderr)
        self.assertIn('原报告保持不变', stderr)
        ocr.assert_not_called()
        parse.assert_not_called()
        self.assertEqual(self.output.read_bytes(), self.old_report)
        self.assertEqual(self.output.stat().st_mtime_ns, mtime)
        return stderr

    def test_missing_later_samples_stop_all_inference_and_preserve_report(self):
        stderr = self.assert_rejected([
            case('valid', 'valid.png'),
            case('missing-a', 'missing-a.png'),
            case('missing-b', 'missing-b.png'),
        ], '缺少图片')
        self.assertIn('missing-a.png', stderr)
        self.assertIn('missing-b.png', stderr)

    def test_duplicate_ids_abort_before_inference(self):
        self.assert_rejected([case('same', 'valid.png'), case('same', 'valid.png')], '重复样本 ID')

    def test_missing_id_aborts_before_inference(self):
        self.assert_rejected([case('', 'valid.png')], '缺少非空字符串 ID')

    def test_corrupt_image_aborts_before_inference(self):
        (self.samples / 'corrupt.png').write_bytes(b'this is not an image')
        self.assert_rejected([case('corrupt', 'corrupt.png')], '图片无法读取')

    def test_empty_set_cannot_overwrite_existing_report(self):
        self.assert_rejected([], '评测集为空')

    def test_explicit_paths_produce_report_when_all_inputs_are_valid(self):
        status, stderr, ocr, parse = self.run_main([case('valid', 'valid.png')])
        self.assertEqual(status, 0)
        self.assertEqual(stderr, '')
        ocr.assert_called_once()
        parse.assert_called_once()
        report = json.loads(self.output.read_text(encoding='utf-8'))
        self.assertEqual(report['samples'], 1)
        self.assertEqual(report['association_accuracy_strict'], 1.0)
        self.assertEqual(report['rows'][0]['id'], 'valid')


if __name__ == '__main__':
    unittest.main()
