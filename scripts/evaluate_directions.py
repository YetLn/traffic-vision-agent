"""Evaluate saved full-image direction predictions; never load a detector or OCR model.

Gold: {metadata: {...}, cases: [{case_id, source_image, signs: [
  {bbox, negative, relations: [{destination, direction, text_bbox?, arrow_bbox?}]}]}]}.
A bare list of cases is also accepted as development data.
Predictions: {case_id: {signs: [{detection_bbox, relations: [
  {destination, direction, text_bbox, arrow_bbox}]}]}}; an optional predictions wrapper is allowed.
All boxes are positive-area [x1, y1, x2, y2] boxes in ORIGINAL IMAGE coordinates.

Usage: python scripts/evaluate_directions.py --gold gold.json --predictions predictions.json
       --output outputs/direction_evaluation.json
"""

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import sys
import unicodedata

from scipy.optimize import linear_sum_assignment


DIRECTIONS = {'up', 'left', 'right', 'slight_left', 'slight_right', 'down'}


class ValidationError(ValueError):
    pass


def normalize_destination(value):
    return ''.join(unicodedata.normalize('NFKC', value).split()).casefold()


def valid_bbox(value, label):
    if not isinstance(value, list) or len(value) != 4:
        raise ValidationError(f'{label}: expected [x1, y1, x2, y2]')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValidationError(f'{label}: coordinates must be finite numbers')
    x1, y1, x2, y2 = value
    if min(value) < 0 or x2 <= x1 or y2 <= y1 or not math.isfinite((x2-x1)*(y2-y1)):
        raise ValidationError(f'{label}: expected nonnegative coordinates and positive finite area')


def relation_key(relation, label, evidence_required):
    if not isinstance(relation, dict):
        raise ValidationError(f'{label}: relation must be an object')
    destination, direction = relation.get('destination'), relation.get('direction')
    if not isinstance(destination, str) or not normalize_destination(destination):
        raise ValidationError(f'{label}: destination must be nonempty text')
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValidationError(f'{label}: unsupported direction {direction!r}')
    for name in ('text_bbox', 'arrow_bbox'):
        if evidence_required or name in relation:
            valid_bbox(relation.get(name), f'{label}.{name}')
    return normalize_destination(destination), direction


def validate_gold(gold):
    if isinstance(gold, list):
        cases, metadata = gold, {}
    elif isinstance(gold, dict):
        cases, metadata = gold.get('cases'), gold.get('metadata', {})
    else:
        raise ValidationError('gold must be a cases list or an object containing cases')
    if not isinstance(cases, list) or not cases:
        raise ValidationError('gold cases must be a nonempty list')
    if not isinstance(metadata, dict):
        raise ValidationError('gold metadata must be an object')
    metadata = dict(metadata)
    metadata.setdefault('split', 'dev')
    metadata.setdefault('entity_reviewed', False)
    metadata.setdefault('frozen', False)
    for name in ('entity_reviewed', 'frozen'):
        if not isinstance(metadata[name], bool):
            raise ValidationError(f'metadata.{name} must be a boolean')
    if metadata['split'] not in ('dev', 'test'):
        raise ValidationError('metadata.split must be dev or test')
    frozen = metadata['split'] == 'test' and metadata['entity_reviewed'] and metadata['frozen']
    if (metadata['split'] == 'test' or metadata['frozen']) and not frozen:
        raise ValidationError('frozen test requires split=test, entity_reviewed=true and frozen=true')
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValidationError(f'gold case {index}: expected an object')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValidationError(f'gold case {index}: missing case_id')
        if case_id != case_id.strip() or case_id in seen:
            raise ValidationError(f'duplicate or whitespace-padded case_id: {case_id!r}')
        seen.add(case_id)
        if not isinstance(case.get('source_image'), str) or not case['source_image'].strip():
            raise ValidationError(f'{case_id}: missing source_image')
        if not isinstance(case.get('signs'), list):
            raise ValidationError(f'{case_id}: signs must be a list')
        boxes = set()
        for sign_index, sign in enumerate(case['signs']):
            label = f'gold {case_id}.signs[{sign_index}]'
            if not isinstance(sign, dict):
                raise ValidationError(f'{label}: expected an object')
            valid_bbox(sign.get('bbox'), label + '.bbox')
            box = tuple(sign['bbox'])
            if box in boxes:
                raise ValidationError(f'{label}: duplicate gold sign bbox')
            boxes.add(box)
            if not isinstance(sign.get('negative'), bool) or not isinstance(sign.get('relations'), list):
                raise ValidationError(f'{label}: negative must be boolean and relations must be a list')
            if sign['negative'] and sign['relations']:
                raise ValidationError(f'{label}: negative sign cannot contain decidable relations')
            if not sign['negative'] and not sign['relations']:
                raise ValidationError(f'{label}: nonnegative sign requires at least one relation')
            keys = set()
            for relation in sign['relations']:
                key = relation_key(relation, label, False)
                if key in keys:
                    raise ValidationError(f'{label}: duplicate gold relation {key}')
                keys.add(key)
    return cases, metadata, frozen


def validate_predictions(predictions, case_ids):
    if isinstance(predictions, dict) and 'predictions' in predictions:
        predictions = predictions['predictions']
    if not isinstance(predictions, dict):
        raise ValidationError('predictions must map case_id to a full-image report')
    extra = set(predictions) - set(case_ids)
    if extra:
        raise ValidationError(f'predictions contain unknown case IDs: {sorted(extra)}')
    for case_id, report in predictions.items():
        if not isinstance(report, dict) or not isinstance(report.get('signs'), list):
            raise ValidationError(f'prediction {case_id}: report must contain a signs list')
        for index, sign in enumerate(report['signs']):
            label = f'prediction {case_id}.signs[{index}]'
            if not isinstance(sign, dict):
                raise ValidationError(f'{label}: expected an object')
            valid_bbox(sign.get('detection_bbox'), label + '.detection_bbox')
            if not isinstance(sign.get('relations'), list):
                raise ValidationError(f'{label}: relations must be a list')
            for relation in sign['relations']:
                relation_key(relation, label, True)
    return predictions


def bbox_iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def match_signs(gold, predicted, threshold):
    """Maximum-cardinality matching, breaking ties by total IoU; labels never affect matching."""
    if not gold or not predicted:
        return []
    overlaps = [[bbox_iou(g['bbox'], p['detection_bbox']) for p in predicted] for g in gold]
    bonus = max(len(gold), len(predicted)) + 1
    costs = [[-(bonus+iou) if iou >= threshold else 0 for iou in row] for row in overlaps]
    gold_indices, predicted_indices = linear_sum_assignment(costs)
    return [(int(g), int(p), overlaps[g][p]) for g, p in zip(gold_indices, predicted_indices)
            if overlaps[g][p] >= threshold]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(gold, predictions, iou_threshold=0.5):
    if not isinstance(iou_threshold, (int, float)) or isinstance(iou_threshold, bool) or not 0 < iou_threshold <= 1:
        raise ValidationError('IoU threshold must be greater than 0 and at most 1')
    cases, metadata, frozen = validate_gold(gold)
    predictions = validate_predictions(predictions, [case['case_id'] for case in cases])
    totals = defaultdict(int)
    evidence = {'text_bbox': [], 'arrow_bbox': []}
    rows = []
    for case in cases:
        case_id, gold_signs = case['case_id'], case['signs']
        predicted = predictions.get(case_id, {'signs': []})['signs']
        matches = match_signs(gold_signs, predicted, iou_threshold)
        counts = {'gold_signs': len(gold_signs), 'predicted_signs': len(predicted),
                  'matched_signs': len(matches),
                  'gold_relations': sum(len(sign['relations']) for sign in gold_signs),
                  'predicted_relations': sum(len(sign['relations']) for sign in predicted),
                  'correct_relations': 0,
                  'negative_signs': sum(sign['negative'] for sign in gold_signs),
                  'negative_signs_with_directions': 0}
        match_rows = []
        for gi, pi, overlap in matches:
            available = defaultdict(deque)
            for relation in gold_signs[gi]['relations']:
                available[relation_key(relation, 'validated gold', False)].append(relation)
            correct = 0
            for relation in predicted[pi]['relations']:
                key = relation_key(relation, 'validated prediction', True)
                if not available[key]:
                    continue
                target = available[key].popleft()
                correct += 1
                for name in evidence:
                    if name in target:
                        evidence[name].append(bbox_iou(target[name], relation[name]))
            counts['correct_relations'] += correct
            match_rows.append({'gold_sign_index': gi, 'prediction_sign_index': pi,
                               'sign_iou': overlap, 'correct_relations': correct})
        # Include unmatched duplicate detections: they must not hide a false assertion on a negative sign.
        for sign in gold_signs:
            if sign['negative'] and any(p['relations'] and bbox_iou(sign['bbox'], p['detection_bbox']) >= iou_threshold
                                        for p in predicted):
                counts['negative_signs_with_directions'] += 1
        counts['false_positive_relations'] = counts['predicted_relations'] - counts['correct_relations']
        counts['missed_gold_relations'] = counts['gold_relations'] - counts['correct_relations']
        for key, value in counts.items():
            totals[key] += value
        rows.append({'case_id': case_id, 'source_image': case['source_image'],
                     'prediction_missing': case_id not in predictions, 'counts': counts, 'matches': match_rows,
                     'unmatched_gold_signs': [i for i in range(len(gold_signs)) if i not in {g for g, _, _ in matches}],
                     'unmatched_prediction_signs': [i for i in range(len(predicted)) if i not in {p for _, p, _ in matches}]})
    return {'evaluation_scope': 'frozen_test' if frozen else 'development_only',
            'notice': ('Frozen test status is based on supplied entity_reviewed/frozen declarations.' if frozen else
                       'Development evaluation only; exploratory results are not a frozen-test generalization claim.'),
            'metadata': metadata,
            'matching': {'sign_iou_threshold': iou_threshold, 'sign_policy': 'one-to-one maximum cardinality, then total IoU',
                         'destination_normalization': 'Unicode NFKC, remove whitespace, casefold; no fuzzy match',
                         'relation_policy': 'one-to-one exact normalized destination and direction within matched signs',
                         'evidence_coordinates': 'original image',
                         'negative_policy': 'each negative gold sign with any relation-emitting prediction at IoU threshold'},
            'case_count': len(cases), 'counts': dict(totals),
            'relation_precision': ratio(totals['correct_relations'], totals['predicted_relations']),
            'relation_coverage': ratio(totals['correct_relations'], totals['gold_relations']),
            'negative_sign_error_rate': ratio(totals['negative_signs_with_directions'], totals['negative_signs']),
            'evidence_on_correct_relations': {
                name: {'compared_count': len(values), 'mean_iou': ratio(sum(values), len(values))}
                for name, values in evidence.items()},
            'cases': rows}


def load_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f'{path}: duplicate JSON key {key!r}')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs'/'direction_evaluation.json')
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_json(args.gold), load_json(args.predictions))
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, OSError) as exc:
        print(f'Evaluation failed; existing report unchanged: {exc}', file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('evaluation_scope', 'relation_precision', 'relation_coverage',
                                                  'negative_sign_error_rate')}, ensure_ascii=False))
    print(f'Report: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
