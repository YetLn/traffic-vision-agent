"""Materialize visual development annotations in original image coordinates.

The hand-entered rectangles below are crop-local approximations read directly
from source pixels, without using detector/OCR predictions. Source images are
never changed. They still require independent human pixel review.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/direction_annotations_v2'

# Each record: evaluation scope, note, board quad, visible text boxes and arrow
# boxes, then scored simple relationships. Quads/boxes are crop-local pixels.
VISUAL={
0: dict(scope='simple_independent_rows', note='完整斜视蓝牌；两条独立箭头分别对应翠园路和弹性道路。',
 quad=[[207,299],[797,2],[861,968],[8,1128]],
 texts=[('翠园路',[368,125,753,357]),('弹性道路',[277,537,758,796])],
 arrows=[('up',[207,349,334,501],'independent_route_arrow'),('left',[106,723,255,898],'independent_route_arrow')],
 relations=[(0,0),(1,1)]),
1: dict(scope='clipped_or_ambiguous_board',note='路牌底端在原图外；景山前街/北长街是道路名，括号内故宫/西长安街与箭头的层级须复核，暂不计正例。',
 quad=[[51,61],[688,1],[681,1357],[29,1357]],
 texts=[('景山前街',[179,99,652,306]),('故宫',[280,454,590,569]),('北长街',[132,654,651,861]),('西长安街',[194,1013,605,1127])],
 arrows=[('left',[63,419,221,552],'independent_route_arrow'),('right',[44,936,204,1078],'independent_route_arrow')],relations=[]),
2: dict(scope='no_independent_arrow',note='垭卓依垭口海拔说明牌，无方向箭头。',
 quad=[[120,11],[880,129],[897,802],[10,810]],
 texts=[('垭卓依垭口',[167,44,823,182]),('海拔',[134,419,347,551]),('4445m',[432,458,848,627])],arrows=[],relations=[]),
3: dict(scope='clipped_no_arrow',note='机场牌下部超出原图，未见独立方向箭头。',
 quad=[[9,1],[836,7],[839,584],[7,584]],texts=[('北京首都机场',[64,331,762,468])],arrows=[],relations=[]),
4: dict(scope='dark_prompt_no_arrow',note='夜间竖牌为前方测速区提示，无独立路线箭头。',
 quad=[[26,1],[193,1],[193,625],[7,628]],texts=[('前方测速区',[37,18,161,575])],arrows=[],relations=[]),
5: dict(scope='connected_road_diagram',note='十字路网图四臂相连，不能拆成独立箭头推断一一对应。',
 quad=[[5,2],[540,7],[540,263],[0,263]],
 texts=[('康居路',[211,16,356,61]),('称山北路',[6,121,200,169]),('康良路',[3,204,197,257]),('悬沙路',[389,205,540,256])],
 arrows=[('unknown',[171,106,409,243],'connected_road_diagram')],relations=[]),
6: dict(scope='pedestrian_direction_stacks',note='三块步行方向条牌；每条完整左箭头对应一个目的地，中文名称不含“方向”后缀。',
 quad=[[43,1],[656,243],[655,525],[0,340]],
 texts=[('金茂大厦',[204,89,514,188]),('环球金融中心',[196,225,619,344]),('上海国金中心',[170,360,620,477])],
 arrows=[('left',[34,37,122,125],'pedestrian_route_arrow'),('left',[29,165,126,267],'pedestrian_route_arrow'),('left',[25,286,122,395],'pedestrian_route_arrow')],
 relations=[(0,0),(1,1),(2,2)]),
7: dict(scope='ambiguous_distance_board',note='白底上箭头与蓝底多条道路和距离的对应范围不唯一，暂列歧义牌。',
 quad=[[20,31],[962,0],[992,400],[0,406]],
 texts=[('解放大道',[54,46,253,139]),('建设大道',[304,125,716,229]),('汉西二路',[298,240,731,357])],
 arrows=[('up',[77,143,195,300],'independent_but_ambiguous_association')],relations=[]),
8: dict(scope='unsupported_diagonal_distance',note='500米后右前方入口，斜向箭头超出直行/左转/右转范围。',
 quad=[[19,23],[995,74],[1025,863],[8,850]],texts=[('G5513',[203,116,504,297]),('G56',[600,137,872,314]),('500m',[186,452,660,691])],
 arrows=[('up_right',[693,471,865,694],'diagonal_route_arrow')],relations=[]),
9: dict(scope='connected_road_diagram',note='三岔方向和道路名围绕同一连通路网图，非独立箭头行。',
 quad=[[10,8],[972,1],[978,470],[3,473]],texts=[('苍山路',[304,46,635,169]),('三茂街',[26,220,301,346]),('兴盛路',[697,222,959,346])],
 arrows=[('unknown',[273,174,620,428],'connected_road_diagram')],relations=[]),
10: dict(scope='distance_no_arrow',note='三行远距里程牌，没有方向箭头。',
 quad=[[10,0],[742,88],[746,624],[4,618]],texts=[('巴什考贡',[99,73,441,183]),('红柳沟',[37,269,464,386]),('和田',[39,473,481,586])],arrows=[],relations=[]),
11: dict(scope='partially_supported_directions',note='上半区一支直行箭头对应武珞路、长江大桥；下半区右前方斜箭头超出范围，不对其输出关系。',
 quad=[[89,58],[560,0],[479,875],[2,879]],
 texts=[('武珞路',[188,78,531,250]),('长江大桥',[188,262,526,413]),('楚善街',[34,504,374,654]),('张之洞路',[29,664,414,819])],
 arrows=[('up',[91,170,168,374],'independent_route_arrow'),('up_right',[367,551,487,767],'diagonal_route_arrow')],
 relations=[(0,0),(1,0)]),
12: dict(scope='connected_road_diagram',note='十字图相连；背景还有其他实体牌，不据此图拆分。',
 quad=[[12,5],[720,2],[728,404],[4,410]],texts=[('越英北路',[226,11,491,91]),('越安北路',[235,91,496,172]),('育贤路',[33,174,233,254]),('孙马公路',[461,177,709,260])],
 arrows=[('unknown',[194,178,552,381],'connected_road_diagram')],relations=[]),
13: dict(scope='lane_guidance',note='车道级连续导向，左转/直行/右转符号属于车道说明，非独立地名行。',
 quad=[[2,56],[299,0],[303,182],[2,232]],texts=[('长夜山路',[179,128,299,193])],
 arrows=[('left',[10,83,59,166],'lane_arrow'),('up',[63,75,104,168],'lane_arrow'),('up',[113,52,149,159],'lane_arrow'),('up',[163,39,198,147],'lane_arrow'),('right',[236,25,291,136],'lane_arrow')],relations=[]),
14: dict(scope='connected_road_diagram',note='直行加右转T形符号连通；单独拆分会虚构指向关系。',
 quad=[[17,125],[302,0],[304,269],[0,353]],texts=[('兰亭凤路',[25,75,171,154]),('开亭路',[22,152,166,239]),('桃江中学',[182,129,301,217])],
 arrows=[('unknown',[62,206,274,344],'connected_road_diagram')],relations=[]),
15: dict(scope='connected_road_diagram_low_light',note='低照度且道路图连通，箭头与文字不满足独立匹配。',
 quad=[[0,47],[516,1],[537,316],[1,342]],texts=[('解放大道',[148,98,298,176]),('绿云中路',[334,252,520,326])],
 arrows=[('unknown',[135,178,401,320],'connected_road_diagram')],relations=[]),
}


def main():
    source=json.loads((OUT/'candidates.json').read_text(encoding='utf-8'))['cases']
    if len(source)!=len(VISUAL) or set(VISUAL)!=set(range(len(source))):
        raise ValueError('Visual records must cover every selected source')
    cases=[]
    checks=OUT/'checks';checks.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for index,item in enumerate(source):
        row=VISUAL[index]
        with Image.open(item['image']) as opened:
            photo=opened.convert('RGB')
        x0,y0,x1,y1=item['bbox'];cw,ch=x1-x0,y1-y0
        def bbox(value):
            x,y,r,b=value
            if not (0<=x<r<=cw and 0<=y<b<=ch):
                raise ValueError(f'Out-of-bounds crop box: {index} {value} {(cw,ch)}')
            return [x+x0,y+y0,r+x0,b+y0]
        def quad(value):
            if any(not (0<=x<=cw and 0<=y<=ch) for x,y in value):
                raise ValueError(f'Out-of-bounds panel polygon: {index}')
            return [[x+x0,y+y0] for x,y in value]
        texts=[{'text_id':f't{i+1}','text':t,'bbox':bbox(b),'role':'destination_or_context',
                'reviewer':'assistant_visual'} for i,(t,b) in enumerate(row['texts'])]
        arrows=[{'arrow_id':f'a{i+1}','direction':d,'bbox':bbox(b),'kind':k,
                 'reviewer':'assistant_visual'} for i,(d,b,k) in enumerate(row['arrows'])]
        relations=[]
        for ti,ai in row['relations']:
            t,a=texts[ti],arrows[ai]
            if a['direction'] not in ('up','left','right'):
                raise ValueError(f'Unsupported scored relation {index}')
            relations.append({'destination':t['text'],'direction':a['direction'],
                              'text_id':t['text_id'],'arrow_id':a['arrow_id'],
                              'group_id':f'g{ai+1}', 'text_bbox':t['bbox'],'arrow_bbox':a['bbox']})
            t['role']='scored_destination'
        sign={'bbox':list(item['bbox']),'polygon':quad(row['quad']),
              'negative':not bool(relations),'relations':relations,
              'texts':texts,'arrows':arrows,'layout':row['scope'],
              'review_notes':row['note'],'physical_board_fully_visible':index not in (1,3),
              'route_arrow_annotation_exhaustive':index in (0,1,2,3,4,6,8,10,11),
              'text_annotation_exhaustive':False,
              'evidence_coordinate_space':'original_image_pixels'}
        case={'case_id':item['case_id'],'sample_id':item['sample_id'],
              'source_image':item['image'],'source_sha256':item['source_sha256'],
              'source_group_id':item['group_id'],'source_split_in_manifest':item['split'],
              'annotation_partition':item['annotation_partition'],'effective_split':'dev',
              'frozen':False,'seen_during_development':True,'entity_reviewed':False,
              'human_verified':False,'reviewer':'assistant_visual','image_size':item['image_size'],
              'annotation_method':'manual visual inspection of source image without detector or OCR predictions',
              'annotation_precision':'approximate boxes and polygons; independent human review needed',
              'signs':[sign]}
        cases.append(case)
        draw=ImageDraw.Draw(photo)
        draw.rectangle(sign['bbox'],outline='yellow',width=5)
        draw.line([tuple(p) for p in sign['polygon']+[sign['polygon'][0]]],fill='orange',width=4)
        for t in texts:
            draw.rectangle(t['bbox'],outline='cyan',width=3)
            draw.text((t['bbox'][0],max(0,t['bbox'][1]-25)),t['text_id'],font=font,fill='cyan')
        for a in arrows:
            draw.rectangle(a['bbox'],outline='magenta',width=3)
            draw.text((a['bbox'][0],max(0,a['bbox'][1]-25)),a['arrow_id'],font=font,fill='magenta')
        check=checks/(item['case_id']+'.jpg')
        photo.save(check,quality=90)
        case['check_image']=str(check.relative_to(OUT))
    metadata={'split':'dev','frozen':False,'human_verified':False,'entity_reviewed':False,
              'reviewer':'assistant_visual','review_date':'2026-09-23',
              'purpose':'Expanded manual visual evidence and separate additional original-image checks; not frozen acceptance',
              'coordinate_frame':'original stored image pixels; no EXIF-rotated cases selected',
              'selection':'16 distinct new source images, six night capture groups, excluding hidden augmented night files',
              'annotation_scope':'one selected target board or board cluster per image; multilingual, auxiliary and secondary image signs not exhaustive',
              'negative_policy':'no arrow, connected map, lane diagram, distance/diagonal/ambiguous or truncated signs should abstain',
              'known_limitations':'box corners are approximate and have not been independently human reviewed; all images are development-exposed'}
    payload={'metadata':metadata,'cases':cases}
    (OUT/'annotations.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for partition in ('development','additional_validation'):
        subset={'metadata':{**metadata,'annotation_partition':partition},
                'cases':[c for c in cases if c['annotation_partition']==partition]}
        (OUT/f'{partition}.gold.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2),encoding='utf-8')
    # Only a wholly visible development board with all independent route arrows
    # exhaustively traced is safe as an arrow-detector training crop. Clipped,
    # negative and additional-validation images are intentionally excluded.
    export=OUT/'arrow_yolo_development_only'
    (export/'images').mkdir(parents=True,exist_ok=True)
    (export/'labels').mkdir(parents=True,exist_ok=True)
    exported=[]
    class_ids={'up':0,'left':1,'right':2}
    for case in cases:
        sign=case['signs'][0]
        if (case['annotation_partition']!='development' or
                sign['layout']!='simple_independent_rows' or
                not sign['physical_board_fully_visible'] or
                not sign['route_arrow_annotation_exhaustive']):
            continue
        name=case['case_id']; x0,y0,x1,y1=sign['bbox'];w,h=x1-x0,y1-y0
        with Image.open(case['source_image']) as image:
            image.convert('RGB').crop(sign['bbox']).save(export/'images'/f'{name}.png')
        labels=[]
        for arrow in sign['arrows']:
            if arrow['kind']!='independent_route_arrow' or arrow['direction'] not in class_ids:
                raise ValueError('Incomplete arrow class in supposedly exhaustive panel')
            ax,ay,ar,ab=arrow['bbox']
            labels.append(f"{class_ids[arrow['direction']]} {((ax+ar)/2-x0)/w:.7f} "
                          f"{((ay+ab)/2-y0)/h:.7f} {(ar-ax)/w:.7f} {(ab-ay)/h:.7f}")
        (export/'labels'/f'{name}.txt').write_text('\n'.join(labels)+'\n',encoding='utf-8')
        exported.append({'case_id':name,'arrow_count':len(labels),'source_sha256':case['source_sha256']})
    (export/'data.yaml').write_text('path: .\ntrain: images\nnames: [up, left, right]\n',encoding='utf-8')
    (export/'manifest.json').write_text(json.dumps({'use':'development-only annotation export; no independent validation split',
        'class_names':['up','left','right'],'samples':exported,'human_verified':False},
        ensure_ascii=False,indent=2),encoding='utf-8')
    print({'images':len(cases),'development':6,'additional_validation':10,
           'board_annotations':len(cases),'text_boxes':sum(len(c['signs'][0]['texts']) for c in cases),
           'arrow_or_diagram_boxes':sum(len(c['signs'][0]['arrows']) for c in cases),
           'scored_relations':sum(len(c['signs'][0]['relations']) for c in cases)})


if __name__=='__main__':main()
