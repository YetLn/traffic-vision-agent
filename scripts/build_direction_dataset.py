"""Read-only source extraction with exact image/label pairing and auditable grouping.

Outputs stay under outputs/direction_dataset_v1. Never joins images by bare basename.
Candidate splits are provisional until entity grouping is reviewed; not a detector benchmark.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {'day': Path(r'D:\Private-dataset-master'),
           'night': Path(r'D:\images-night\from hainan\imagesnight')}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def night_exclusion_reason(image, metadata):
    """Manifest blanks do not prove an image is an original capture."""
    if not metadata:
        return 'untracked'
    if str(metadata.get('sample_type', '')).strip():
        return 'manifest_augmentation'
    provenance = ' '.join(str(value).lower() for value in
                          (image, metadata.get('source_image', ''), metadata.get('original_stem', '')))
    if any(marker in provenance for marker in
           ('light_attack', '_existing_source_attack', '_global_side_height_radius')):
        return 'augmentation_provenance'
    return None


def pairs(root):
    # The private dataset's validation/test label folders are empty. A fallback is
    # permitted only when the original label index has exactly ONE matching stem.
    fallback={}
    for path in sorted((root/'labels').rglob('*.txt')) if (root/'labels').exists() else []:
        fallback.setdefault(path.stem,[]).append(path)
    folders = [(root/'images', root/'labels')]
    folders += [(root/s/'images', root/s/'labels') for s in ('train','val','test')]
    for images, labels in folders:
        if not images.exists():
            continue
        for image in sorted(images.rglob('*')):
            if image.suffix.lower() in EXTENSIONS and image.is_file():
                label = labels/image.relative_to(images).with_suffix('.txt')
                if not label.exists() and len(fallback.get(image.stem,[]))==1:
                    label=fallback[image.stem][0]
                yield image, label


def read_boxes(path, issues=None):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError('expected detection labels with five columns')
        values = [float(x) for x in fields]
        cls, x, y, w, h = values
        if not all(np.isfinite(values)) or cls != int(cls) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            if issues is None:
                raise ValueError('invalid normalized detection box')
            issues.append({'label':str(path),'line':line_no,'error':'invalid normalized detection box; row skipped'})
            continue
        if int(cls) == 2:
            rows.append((line_no, x, y, w, h))
    return rows


def phash(image):
    gray = np.asarray(image.convert('L').resize((32,32)), dtype=np.float32)
    frequencies = cv2.dct(gray)[:8,:8].flatten()[1:]
    value = 0
    for bit in frequencies > np.median(frequencies):
        value = (value << 1) | int(bit)
    return value


def contact_sheet(rows, out, prefix):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    for page in range(0,len(rows),20):
        selected = rows[page:page+20]
        sheet = Image.new('RGB',(1400,5*230),'#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for index,row in enumerate(selected):
            x,y=(index%4)*350,(index//4)*230
            with Image.open(out/row['crop']) as image:
                image.thumbnail((338,195))
                sheet.paste(image,(x+6,y+4))
            draw.text((x+6,y+203),f"{row['sample_id']} {row['crop_size'][0]}x{row['crop_size'][1]}",font=font,fill='black')
        sheet.save(out/f'{prefix}_{page//20+1:02}.jpg',quality=90)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/direction_dataset_v1')
    parser.add_argument('--per-source',type=int,default=140)
    args=parser.parse_args()
    out=args.out
    if (out/'manifest.json').exists():
        raise SystemExit('Manifest already exists; use a new --out to avoid changing frozen records.')
    (out/'crops').mkdir(parents=True,exist_ok=True)
    stats=Counter(); candidates=[]; issues=[]
    for source,root in SOURCES.items():
        metadata={}
        if source=='night':
            with (root/'image_level_manifest.csv').open(encoding='utf-8-sig',newline='') as handle:
                metadata={str(Path(row['image']).resolve()):row for row in csv.DictReader(handle)}
        for image,label in pairs(root):
            stats[f'{source}_images']+=1
            meta=metadata.get(str(image.resolve()),{})
            exclusion=night_exclusion_reason(image,meta) if source=='night' else None
            if exclusion:
                stats['night_augmented_or_untracked_skipped']+=1
                stats[f'night_excluded_{exclusion}']+=1
                continue
            if not label.exists():
                stats[f'{source}_missing_label']+=1
                continue
            try:
                boxes=read_boxes(label, issues)
                if not boxes:
                    continue
                with Image.open(image) as handle:
                    width,height=handle.size
                    orientation=handle.getexif().get(274,1)
                if orientation != 1:
                    stats['skipped_exif_orientation']+=1
                    continue
                stats[f'{source}_point_l_images']+=1
                for line_no,x,y,w,h in boxes:
                    # Crop in the stored-pixel frame used by YOLO labels; no silent rotation.
                    box=[max(0,int((x-w/2)*width)),max(0,int((y-h/2)*height)),
                         min(width,int(np.ceil((x+w/2)*width))),min(height,int(np.ceil((y+h/2)*height)))]
                    bw,bh=box[2]-box[0],box[3]-box[1]
                    if min(bw,bh)<96 or bw*bh<20000:
                        continue
                    candidates.append({'source':source,'image':str(image),'label':str(label),
                        'capture_group':meta.get('group_id') or None,'lighting_level':meta.get('level') or None,
                        'label_line':line_no,'source_split':image.relative_to(root).parts[0],
                        'bbox':box,'image_size':[width,height],'crop_size':[bw,bh],
                        'area':bw*bh})
            except (ValueError,OSError) as exc:
                issues.append({'image':str(image),'label':str(label),'error':str(exc)})
    selected=[]
    for source in SOURCES:
        rows=sorted([r for r in candidates if r['source']==source],key=lambda r:(-r['area'],r['image'],r['label_line']))
        selected.extend(rows[:args.per_source])
    samples=[]; duplicates=[]; seen={}; image_hashes={}
    for row in selected:
        path=Path(row['image'])
        if path not in image_hashes:
            image_hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
        sha=image_hashes[path]
        identity=(sha,tuple(row['bbox']))
        if identity in seen:
            duplicates.append({'source_image':row['image'],'duplicate_of':seen[identity]})
            continue
        crop_identity=hashlib.sha256(json.dumps(row['bbox']).encode()).hexdigest()[:8]
        sid=f"{row['source']}-{sha[:10]}-{crop_identity}"
        with Image.open(path) as image:
            crop=image.convert('RGB').crop(row['bbox'])
            crop_path=Path('crops')/f'{sid}.png'
            crop.save(out/crop_path)
            p=phash(crop)
        samples.append({**row,'sample_id':sid,'source_sha256':sha,
                        'label_sha256':hashlib.sha256(Path(row['label']).read_bytes()).hexdigest(),
                        'crop':crop_path.as_posix(),'crop_sha256':hashlib.sha256((out/crop_path).read_bytes()).hexdigest(),
                        'phash':f'{p:016x}'})
        seen[identity]=sid
    # Conservative near-duplicate grouping. Not a substitute for entity review.
    parent=list(range(len(samples)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(samples):
        for j,b in enumerate(samples[:i]):
            aspect_a=a['crop_size'][0]/a['crop_size'][1];aspect_b=b['crop_size'][0]/b['crop_size'][1]
            near=(int(a['phash'],16)^int(b['phash'],16)).bit_count()<=10 and abs(np.log(aspect_a/aspect_b))<0.25
            same_capture=a['source']==b['source']=='night' and a.get('capture_group') and a.get('capture_group')==b.get('capture_group')
            if a['source_sha256']==b['source_sha256'] or same_capture or near:
                parent[find(i)]=find(j)
    groups={}
    for i,row in enumerate(samples): groups.setdefault(find(i),[]).append(row)
    representatives=[]
    for rows in groups.values():
        gid=min(r['sample_id'] for r in rows)
        split='development' if int(hashlib.sha256(gid.encode()).hexdigest()[:8],16)%10<3 else 'holdout_candidate'
        for row in rows: row.update(group_id=gid,split=split,entity_reviewed=False)
        representatives.append(max(rows,key=lambda r:r['area']))
    manifest={'schema_version':1,'selection':'largest point-l label crops, minimum side 96 and area 20000; selection bias acknowledged',
        'pairing':'same split and relative path; unique-only original label fallback for missing split labels',
        'night_policy':'exclude augmented/untracked rows plus attack filename/source provenance; union full capture groups before splitting',
        'split_status':'provisional; entity groups and gold annotations need review before accuracy claims',
        'detector_evaluation_warning':'source images may have trained existing YOLO; not an unbiased detector benchmark',
        'stats':dict(stats),'eligible_crops':len(candidates),'samples':samples,'issues':issues,
        'exact_duplicates_skipped':duplicates,'group_count':len(groups)}
    payload=json.dumps(manifest,ensure_ascii=False,indent=2)
    (out/'manifest.json').write_text(payload,encoding='utf-8')
    (out/'manifest.sha256').write_text(hashlib.sha256(payload.encode()).hexdigest()+'\n',encoding='ascii')
    for split in ('development','holdout_candidate'):
        rows=[r for r in representatives if r['split']==split]
        contact_sheet(rows,out,split)
    annotations=[{'sample_id':r['sample_id'],'group_id':r['group_id'],'split':r['split'],
        'review_status':'pending','transcript':None,'layout':None,'arrows':[],'routes':[],
        'entity_group_confirmed':False,'notes':''} for r in representatives]
    (out/'annotations.pending.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'stats':stats,'eligible':len(candidates),'extracted':len(samples),'groups':len(groups),
        'representatives':dict(Counter(r['split'] for r in representatives)),
        'source_counts':dict(Counter(r['source'] for r in samples)),'issues':len(issues),'output':str(out)},ensure_ascii=False))


if __name__=='__main__': main()
