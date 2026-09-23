"""Run the preserved v2 source on additional ORIGINAL road images.

Uses the same installed libraries/weights as v3, but imports the immutable v2
package copy. Gold is used only to obtain source_image and case_id, never boxes.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'outputs/direction_upgrade_v3/baseline_source'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((SNAPSHOT/'manifest.json').read_text(encoding='utf-8'))
    if any(hashlib.sha256((SNAPSHOT/name).read_bytes()).hexdigest()!=sha for name,sha in manifest.items()):
        raise RuntimeError('Baseline source snapshot hash mismatch')
    cases=json.loads(args.gold.read_text(encoding='utf-8'))['cases']
    ids=[case['case_id'] for case in cases]
    if not ids or any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',v) for v in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Invalid or duplicate case IDs')
    for case in cases:
        with Image.open(case['source_image']) as im: im.verify()
    sys.path.insert(0,str(SNAPSHOT))
    from traffic_agent import config
    config.ROOT=ROOT
    config.WEIGHTS=ROOT/'weights/traffic_best.pt'
    config.OUTPUTS=ROOT/'outputs'
    sys.path.insert(0,str(ROOT/'vendor'))
    from traffic_agent.detector import TrafficSignDetector
    from traffic_agent.direction_reader import read_directions,draw_evidence
    detector=TrafficSignDetector(weight_path=config.WEIGHTS)
    args.out.mkdir(parents=True,exist_ok=True)
    predictions={}
    for case in cases:
        with Image.open(case['source_image']) as im:
            report=read_directions(im,detector)
            draw_evidence(im,report,args.out/(case['case_id']+'.jpg'))
        report['source_sha256']=hashlib.sha256(Path(case['source_image']).read_bytes()).hexdigest()
        predictions[case['case_id']]=report
        print(case['case_id'],'signs',len(report['signs']),'relations',sum(len(s['relations']) for s in report['signs']),flush=True)
    result={'run_type':'preserved_v2_original_image_yolo','source_snapshot':str(SNAPSHOT),
            'implementation_sha256':manifest,'weights_sha256':hashlib.sha256(config.WEIGHTS.read_bytes()).hexdigest(),
            'predictions':predictions}
    (args.out/'predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
