"""Explicit live API smoke test. Loads local credentials without printing them."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from PIL import Image
from traffic_agent.config import ROOT
from traffic_agent.agent import ask
from traffic_agent.detector import TrafficSignDetector


class SyntheticDetector:
    names = {0:'wran', 1:'ban', 2:'point-l', 3:'point-s'}

    def detect(self, image, conf):
        return {'total':2, 'counts':{'ban':2}, 'detections':[
            {'class_id':1, 'class_name':'ban', 'confidence':0.8, 'bbox':[1,1,4,4]},
            {'class_id':1, 'class_name':'ban', 'confidence':0.7, 'bbox':[5,5,9,9]}]}

    def visualize(self, image, detections, path):
        image.save(path)
        return str(path)


def main():
    # Default uses fabricated data only, never reads private images or weights.
    real_images = '--real-images' in sys.argv
    detector = TrafficSignDetector() if real_images else SyntheticDetector()
    state = None
    records = []
    with (Image.open(ROOT/'test_images/night_01.jpg') if real_images else Image.new('RGB',(10,10))) as image:
        for question in ['有多少个 ban？', '把它们框出来。']:
            result = ask(detector, image, question, mode='DeepSeek Tool Calling', state=state)
            state = result['state']
            assert result['trace'], 'No successful tool call'
            records.append({k:v for k,v in result.items() if k != 'state'})
    assert any(t['tool'] == 'count' for t in records[0]['trace'])
    assert any(t['tool'] == 'visualize' for t in records[1]['trace'])
    assert Path(records[1]['image']).is_file()
    report = {'provider':'DeepSeek', 'model':os.environ.get('DEEPSEEK_MODEL'),
              'data_source':'local image' if real_images else 'fabricated test data; no image transmitted',
              'status':'passed', 'turns':records}
    (ROOT/'outputs/deepseek_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'failed', 'error_type':type(exc).__name__,
                          'http_status':getattr(exc,'status_code',None)}, ensure_ascii=False))
        sys.exit(1)
