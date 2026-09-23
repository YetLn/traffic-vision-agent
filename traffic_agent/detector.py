from collections import Counter
import hashlib
import threading
import time

from PIL import Image, ImageOps, ImageDraw, ImageFont

from .config import WEIGHTS


def image_key(image, conf):
    image = ImageOps.exif_transpose(image).convert('RGB')
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return f'{image.size}:{digest}:{conf:.6f}'


class TrafficSignDetector:
    """One shared model, serialized inference, session-owned JSON results."""

    def __init__(self, weight_path=WEIGHTS, device='cpu'):
        from ultralytics import YOLO
        if not weight_path.is_file():
            raise FileNotFoundError(f'找不到权重：{weight_path}')
        self.model = YOLO(str(weight_path), task='detect')
        self.names = dict(self.model.names)
        self.device = device
        self.lock = threading.Lock()

    def detect(self, image, conf=0.25):
        if not 0 < conf <= 1:
            raise ValueError('置信度阈值必须在 (0, 1] 内。')
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.width * image.height > 40_000_000:
            raise ValueError('请使用不超过 4000 万像素的图片。')
        start = time.perf_counter()
        with self.lock:
            result = self.model.predict(image, conf=conf, imgsz=640,
                                        device=self.device, verbose=False)[0]
            rows = result.boxes.data.cpu().tolist()
        detections = [dict(class_id=int(row[5]), class_name=self.names[int(row[5])],
                           confidence=round(row[4], 4),
                           bbox=[round(x, 2) for x in row[:4]]) for row in rows]
        return dict(detections=detections, total=len(detections),
                    counts=dict(Counter(d['class_name'] for d in detections)),
                    image_size=list(image.size), threshold=conf,
                    elapsed_ms=round((time.perf_counter()-start)*1000, 1))

    @staticmethod
    def visualize(image, detections, output_path):
        image = ImageOps.exif_transpose(image).convert('RGB').copy()
        draw = ImageDraw.Draw(image)
        colors = ['#fbbf24', '#f87171', '#38bdf8', '#4ade80']
        width = max(2, image.width // 600)
        font = ImageFont.load_default(size=max(16, image.width // 120))
        for d in detections:
            box = d['bbox']
            color = colors[d['class_id'] % len(colors)]
            draw.rectangle(box, outline=color, width=width)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            x, y = box[0], max(0, box[1] - font.size - 6)
            bounds = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bounds, fill=color)
            draw.text((x, y), label, fill='black', font=font)
        image.save(output_path)
        return str(output_path)
