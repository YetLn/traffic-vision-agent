import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_directions import ValidationError, evaluate, main


def relation(destination='A地', direction='up', evidence=True):
    result = {'destination': destination, 'direction': direction}
    if evidence:
        result.update(text_bbox=[10, 10, 30, 20], arrow_bbox=[40, 10, 50, 20])
    return result


def gold_sign(negative=False, bbox=None, relations=None):
    return {'bbox': bbox or [0, 0, 100, 100], 'negative': negative,
            'relations': ([] if negative else [relation()]) if relations is None else relations}


def prediction(bbox=None, relations=None):
    return {'detection_bbox': bbox or [0, 0, 100, 100],
            'relations': [relation()] if relations is None else relations}


def gold(signs=None):
    return [{'case_id': 'image-a', 'source_image': 'images/a.jpg',
             'signs': [gold_sign()] if signs is None else signs}]


class DirectionEvaluationTests(unittest.TestCase):
    def test_correct_relation_normalization_and_evidence(self):
        result = evaluate(gold(), {'image-a': {'signs': [prediction(relations=[relation(' ａ 地 ')])]}})
        self.assertEqual(result['relation_precision'], 1)
        self.assertEqual(result['relation_coverage'], 1)
        self.assertEqual(result['evaluation_scope'], 'development_only')
        self.assertEqual(result['evidence_on_correct_relations']['text_bbox'], {'compared_count': 1, 'mean_iou': 1})

    def test_wrong_direction_and_unmatched_detection_are_false_positives(self):
        result = evaluate(gold(), {'image-a': {'signs': [
            prediction(relations=[relation(direction='right')]), prediction(bbox=[200, 0, 300, 100])]}})
        self.assertEqual(result['counts']['false_positive_relations'], 2)
        self.assertEqual(result['relation_precision'], 0)
        self.assertEqual(result['relation_coverage'], 0)

    def test_missing_case_counts_as_missed_relations(self):
        result = evaluate(gold(), {})
        self.assertIsNone(result['relation_precision'])
        self.assertEqual(result['relation_coverage'], 0)
        self.assertEqual(result['counts']['missed_gold_relations'], 1)
        self.assertTrue(result['cases'][0]['prediction_missing'])

    def test_empty_denominators_are_null(self):
        result = evaluate(gold([]), {'image-a': {'signs': []}})
        self.assertIsNone(result['relation_precision'])
        self.assertIsNone(result['relation_coverage'])
        self.assertIsNone(result['negative_sign_error_rate'])

    def test_duplicate_predictions_do_not_inflate_correct_count(self):
        result = evaluate(gold(), {'image-a': {'signs': [
            prediction(relations=[relation(), relation()]), prediction()]}})
        self.assertEqual(result['counts']['correct_relations'], 1)
        self.assertEqual(result['counts']['false_positive_relations'], 2)
        self.assertAlmostEqual(result['relation_precision'], 1/3)

    def test_negative_sign_counts_unmatched_duplicate_direction_output(self):
        result = evaluate(gold([gold_sign(negative=True)]), {'image-a': {'signs': [
            prediction(relations=[]), prediction(bbox=[1, 0, 101, 100])]}})
        self.assertEqual(result['counts']['negative_signs_with_directions'], 1)
        self.assertEqual(result['negative_sign_error_rate'], 1)
        self.assertEqual(result['counts']['false_positive_relations'], 1)

    def test_sign_matching_is_one_to_one_and_uses_iou_cutoff(self):
        second = gold_sign(bbox=[50, 0, 150, 100], relations=[relation('B地')])
        result = evaluate(gold([gold_sign(), second]), {'image-a': {'signs': [prediction(bbox=[25, 0, 125, 100])]}})
        self.assertEqual(result['counts']['matched_signs'], 1)
        self.assertLessEqual(result['counts']['correct_relations'], 1)
        result = evaluate(gold(), {'image-a': {'signs': [prediction(bbox=[60, 0, 160, 100])]}})
        self.assertEqual(result['counts']['matched_signs'], 0)
        self.assertEqual(result['relation_coverage'], 0)

    def test_sign_matching_maximizes_match_count_before_total_iou(self):
        # The best individual overlap A->P1 would strand B. The correct assignment
        # keeps two eligible pairs A->P2 and B->P1, without inspecting their labels.
        signs = [gold_sign(), gold_sign(bbox=[35, 0, 135, 100], relations=[relation('B地')])]
        predictions = [prediction(bbox=[10, 0, 110, 100], relations=[relation('B地')]),
                       prediction(bbox=[0, 0, 60, 100])]
        result = evaluate(gold(signs), {'image-a': {'signs': predictions}})
        self.assertEqual(result['counts']['matched_signs'], 2)
        self.assertEqual(result['counts']['correct_relations'], 2)

    def test_duplicate_gold_cases_and_relations_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, 'case_id'):
            evaluate(gold() + gold(), {})
        with self.assertRaisesRegex(ValidationError, 'duplicate gold relation'):
            evaluate(gold([gold_sign(relations=[relation(), relation(' A 地 ')])]), {})
        with self.assertRaisesRegex(ValidationError, 'duplicate gold sign'):
            evaluate(gold([gold_sign(), gold_sign()]), {})

    def test_invalid_gold_negative_or_bbox_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, 'negative sign'):
            evaluate(gold([gold_sign(negative=True, relations=[relation()])]), {})
        with self.assertRaisesRegex(ValidationError, 'positive finite area'):
            evaluate(gold([gold_sign(bbox=[0, 0, 0, 1])]), {})

    def test_predicted_relations_require_valid_evidence(self):
        for field, value in [('text_bbox', None), ('arrow_bbox', [0, 0, float('nan'), 4]),
                             ('text_bbox', [-1, 0, 2, 4]), ('arrow_bbox', [0, 0, 0, 4])]:
            item = relation()
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                evaluate(gold(), {'image-a': {'signs': [prediction(relations=[item])]}})
        with self.assertRaises(ValidationError):
            evaluate(gold(), {'image-a': {'signs': [prediction(relations=[relation(evidence=False)])]}})

    def test_evidence_iou_is_separate_from_relation_precision(self):
        item = relation()
        item['arrow_bbox'] = [60, 60, 70, 70]
        result = evaluate(gold(), {'image-a': {'signs': [prediction(relations=[item])]}})
        self.assertEqual(result['relation_precision'], 1)
        self.assertEqual(result['evidence_on_correct_relations']['arrow_bbox']['mean_iou'], 0)

    def test_frozen_claim_requires_both_review_and_freeze(self):
        for metadata in ({'split': 'test'}, {'split': 'test', 'frozen': True}, {'split': 'dev', 'frozen': True}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValidationError, 'frozen test'):
                evaluate({'metadata': metadata, 'cases': gold()}, {})
        result = evaluate({'metadata': {'split': 'test', 'entity_reviewed': True, 'frozen': True}, 'cases': gold()}, {})
        self.assertEqual(result['evaluation_scope'], 'frozen_test')

    def test_unknown_prediction_case_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, 'unknown case'):
            evaluate(gold(), {'unknown': {'signs': []}})

    def test_cli_invalid_input_preserves_report_and_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path, pred_path, report_path = root/'gold.json', root/'predictions.json', root/'report.json'
            gold_path.write_text(json.dumps(gold()), encoding='utf-8')
            pred_path.write_text('{"image-a": {"signs": []}, "image-a": {"signs": []}}', encoding='utf-8')
            report_path.write_text('existing report', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                status = main(['--gold', str(gold_path), '--predictions', str(pred_path), '--output', str(report_path)])
            self.assertEqual(status, 2)
            self.assertEqual(report_path.read_text(encoding='utf-8'), 'existing report')


if __name__ == '__main__':
    unittest.main()
