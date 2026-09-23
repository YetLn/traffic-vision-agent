"""Exercise the real local Gradio direction endpoint without any external LLM."""
import json
from pathlib import Path
from gradio_client import Client, handle_file


def main():
    client = Client('http://127.0.0.1:7860', verbose=False)
    answer, evidence, report = client.predict(
        handle_file(r'D:\Private-dataset-master\val\images\08839.jpg'),
        .25, api_name='/read_directions')
    assert report['detector']['entry'] == 'original_image_yolo'
    assert report['coordinate_frame'] == 'exif_normalized_original_pixels'
    assert report['method'] == 'evidence_direction_rectified_v3'
    assert Path(evidence).is_file()
    assert report['signs'] and report['status'] == 'ok'
    relations=[r for s in report['signs'] for r in s['relations']]
    assert {(r['destination'],r['direction']) for r in relations} == {
        ('五里冲路','up'),('北京西路','up'),('观山湖区','up'),('白云','up')}
    assert len(relations)==4 and len({r['arrow_id'] for r in relations})==1
    assert report['detector']['diagnostics']['geometry_only_count']==0
    width, height = report['image_size']
    for sign in report['signs']:
        for item in sign['texts'] + sign['arrows']:
            x,y,r,b = item['bbox']
            assert 0 <= x < r <= width and 0 <= y < b <= height
    negative_answer, negative_image, negative_report = client.predict(
        handle_file(r'D:\Private-dataset-master\test\images\08538.jpg'),.25,api_name='/read_directions')
    assert negative_report['status']=='abstain'
    assert not any(s['relations'] for s in negative_report['signs'])
    assert Path(negative_image).is_file()
    payload = {'transport_verified': True, 'development_examples_passed':True,'semantic_acceptance_passed': False,
               'notice': 'Real local UI run; one shared-arrow example and a no-arrow example, not frozen acceptance.',
               'endpoint': '/read_directions', 'summary': answer, 'report': report,
               'negative_summary':negative_answer,'negative_report':negative_report}
    path = Path(__file__).resolve().parents[1]/'outputs/direction_upgrade_v3/ui_validation.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Local UI: four places share one arrow; no-arrow sign abstains; original-image evidence verified.')


if __name__ == '__main__': main()
