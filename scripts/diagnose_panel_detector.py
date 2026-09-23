"""Development-only full-image panel localization diagnostics; no OCR or LLM.

Gold boxes are used only AFTER inference to calculate localization metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from traffic_agent.config import WEIGHTS
from traffic_agent.detector import TrafficSignDetector
from traffic_agent.panel_detector import detect_panels
from scripts.evaluate_directions import bbox_iou, validate_gold


def overlap_coverage(gold, predicted):
    inter = max(0,min(gold[2],predicted[2])-max(gold[0],predicted[0]))*max(0,min(gold[3],predicted[3])-max(gold[1],predicted[1]))
    return inter/((gold[2]-gold[0])*(gold[3]-gold[1]))


def draw(image, report, path):
    image = ImageOps.exif_transpose(image).convert('RGB').copy()
    painter = ImageDraw.Draw(image)
    for detection in report['diagnostics']['original_yolo_detections']:
        if detection['class_id']==2:
            painter.rectangle(detection['bbox'],outline='red',width=3)
    for index,detection in enumerate(report['detections'],1):
        color = 'lime' if detection['yolo_supported'] else 'orange'
        painter.rectangle(detection['bbox'],outline=color,width=5)
        painter.text((detection['bbox'][0],detection['bbox'][1]),
                     f"{index}: {detection['proposal_source']}",fill=color,stroke_width=1,stroke_fill='black')
    image.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold',type=Path,default=ROOT/'outputs/direction_review/gold.development.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/panel_diagnostics_v1')
    parser.add_argument('--allow-geometry-only',action='store_true')
    args=parser.parse_args(argv)
    payload=json.loads(args.gold.read_text(encoding='utf-8'))
    cases,metadata,frozen=validate_gold(payload)
    if frozen or metadata.get('split')!='dev':
        raise ValueError('This exploratory diagnostic only accepts a development manifest')
    for case in cases:
        with Image.open(case['source_image']) as image:
            image.verify()
    detector=TrafficSignDetector()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,case in enumerate(cases,1):
        with Image.open(case['source_image']) as image:
            report=detect_panels(image,detector,allow_geometry_only=args.allow_geometry_only)
            safe_id=hashlib.sha256(case['case_id'].encode()).hexdigest()[:10]
            evidence=args.out/f'{index:02d}_{safe_id}.jpg'
            draw(image,report,evidence)
        original=[d for d in report['diagnostics']['original_yolo_detections'] if d['class_id']==2]
        evaluations=[]
        for sign in case['signs']:
            box=sign['bbox']
            evaluations.append({'gold_bbox':box,
                'original_best_iou':max((bbox_iou(box,p['bbox']) for p in original),default=0),
                'panel_best_iou':max((bbox_iou(box,p['bbox']) for p in report['detections']),default=0),
                'panel_gold_coverage':max((overlap_coverage(box,p['bbox']) for p in report['detections']),default=0)})
        rows.append({'case_id':case['case_id'],'source_image':case['source_image'],
                     'source_sha256':hashlib.sha256(Path(case['source_image']).read_bytes()).hexdigest(),
                     'evidence':str(evidence),'localization':evaluations,'report':report})
        print(f"{index}/{len(cases)} {Path(case['source_image']).name}: original={len(original)} panels={report['total']} "
              f"IoU={[round(r['panel_best_iou'],3) for r in evaluations]}",flush=True)
    observations=[evaluation for row in rows for evaluation in row['localization']]
    result={'evaluation_scope':'development_only','metadata':metadata,
            'notice':'Exploratory localization only; not direction accuracy and not frozen-test performance.',
            'weights_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
            'allow_geometry_only':args.allow_geometry_only,
            'summary':{'gold_panels':len(observations),
                'original_yolo_iou50_hits':sum(row['original_best_iou']>=.5 for row in observations),
                'fused_panel_iou50_hits':sum(row['panel_best_iou']>=.5 for row in observations),
                'fused_panel_coverage90_hits':sum(row['panel_gold_coverage']>=.9 for row in observations),
                'output_panels':sum(row['report']['total'] for row in rows),
                'geometry_only_outputs':sum(row['report']['diagnostics']['geometry_only_count'] for row in rows)},
            'rows':rows}
    (args.out/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['summary']),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
