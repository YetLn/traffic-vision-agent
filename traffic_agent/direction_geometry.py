"""Conservative, experimental orientation of *standalone* filled arrows.

Directions describe the arrow pixels, not driving instructions. In particular,
``down`` does not mean U-turn. Text position never determines direction. Bent,
diagonal, connected road diagrams and unrecognizable shapes are unsupported.

This is a template-shape baseline, not a trained detector. ``shape_iou`` and
``direction_margin`` are geometric diagnostics, not calibrated probabilities.
Pass OCR boxes to suppress text; missing OCR can still produce false positives.
"""

from collections import Counter
from functools import lru_cache
import hashlib
from pathlib import Path

import cv2
import numpy as np


DIRECTIONS = ("up", "left", "down", "right")
NORMAL_SIZE = 64
_BIT_COUNTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _tight(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _normal(mask):
    return cv2.resize(mask.astype(np.uint8), (NORMAL_SIZE, NORMAL_SIZE),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


@lru_cache(maxsize=1)
def _shape_bank():
    """Triangle heads and notched chevron heads, each pointing up initially."""
    templates, labels = [], []
    size = 96
    for head in (0.30, 0.40, 0.50, 0.60, 0.75, 0.90):
        for shaft in (0.24, 0.34, 0.44):
            variants = [(0.0, 0.0), (0.14, 0.0), (0.26, 0.0), (0.16, 0.18), (0.24, 0.24)]
            if head >= 0.75:
                variants.extend([(0.32, 0.40), (0.40, 0.45)])
            for notch, shoulder in variants:
                if head > 0.60 and shoulder == 0:
                    continue  # A nearly triangular shape is not enough evidence.
                left, right = (1 - shaft) / 2, (1 + shaft) / 2
                # The nine-corner variant has a thick V-shaped head (common
                # on road signs), rather than a filled triangle on a stem.
                points = [(0.5, 0), (1, head - shoulder), (1, head),
                          (right, head - notch), (right, 1), (left, 1),
                          (left, head - notch), (0, head), (0, head - shoulder)]
                base = np.zeros((size, size), np.uint8)
                cv2.fillPoly(base, [np.rint(np.array(points) * (size - 1)).astype(np.int32)], 1)
                for turn, direction in enumerate(DIRECTIONS):
                    oriented = np.rot90(base, turn).copy()
                    for angle in (-10, -5, 0, 5, 10):
                        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
                        matrix[:, 2] += size / 4
                        rotated = cv2.warpAffine(oriented, matrix, (size * 3 // 2, size * 3 // 2),
                                                 flags=cv2.INTER_NEAREST)
                        templates.append(_normal(_tight(rotated)))
                        labels.append(direction)
    array = np.stack(templates).reshape(len(templates), -1)
    return np.packbits(array, axis=1), array.sum(axis=1), np.array(labels)


def classify_arrow_mask(mask, min_iou=0.82, min_margin=0.16):
    """Classify one binary component; return ``direction=None`` on abstention.

    Input must describe one whole foreground object, not an arbitrary window
    around an arrowhead. A tight crop is safe; cutting a road diagram into pieces
    to satisfy this classifier is not.
    """
    foreground = np.asarray(mask) > 0
    if foreground.ndim != 2:
        raise ValueError("An arrow component must be a two-dimensional mask")
    component = _tight(foreground)
    empty = {"direction": None, "shape_iou": 0.0, "direction_margin": 0.0}
    if component is None or min(component.shape) < 9 or int(component.sum()) < 80:
        return {**empty, "reason": "too_small_or_empty"}
    count, _, _, _ = cv2.connectedComponentsWithStats(component.astype(np.uint8), connectivity=8)
    if count != 2:
        return {**empty, "reason": "multiple_components"}
    fill = float(component.mean())
    if not 0.25 <= fill <= 0.82:
        return {**empty, "reason": "unsupported_fill"}
    height, width = component.shape
    ratio = max(height, width) / min(height, width)
    if ratio > 5:
        return {**empty, "reason": "unsupported_aspect"}
    contours, hierarchy = cv2.findContours(component.astype(np.uint8), cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
    hole_area = sum(cv2.contourArea(c) for c, h in zip(contours, hierarchy[0]) if h[3] >= 0)
    if hole_area > 0.06 * component.sum():
        return {**empty, "reason": "hollow_shape"}
    templates, areas, labels = _shape_bank()
    normalized = _normal(component)
    packed = np.packbits(normalized.ravel())
    intersection = _BIT_COUNTS[np.bitwise_and(templates, packed)].sum(axis=1)
    union = areas + normalized.sum() - intersection
    values = intersection / np.maximum(union, 1)
    scores = {direction: float(values[labels == direction].max()) for direction in DIRECTIONS}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    margin = scores[best] - scores[ranked[1]]
    result = {"direction": None, "shape_iou": round(scores[best], 4),
              "direction_margin": round(margin, 4),
              "direction_scores": {name: round(value, 4) for name, value in scores.items()}}
    # Short road arrows can be almost square or have a head wider than their
    # shaft is long. Only a strongly elongated, contradictory axis is invalid.
    if ratio >= 1.8 and (height > width) != (best in ("up", "down")):
        return {**result, "reason": "axis_disagreement"}
    if scores[best] < min_iou:
        return {**result, "reason": "shape_mismatch"}
    if margin < min_margin:
        return {**result, "reason": "ambiguous_orientation"}
    return {**result, "direction": best, "reason": "standalone_arrow_shape"}


def _load_rgb(image):
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return np.asarray(opened.convert("RGB"))
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image array must contain RGB channels")
    return array.astype(np.uint8)


def _overlap(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def _text_coverage(component, bbox, boxes):
    """Measure coverage without removing any foreground or changing topology.

    OCR boxes often include blank margins. A box touching a thin tail should
    not have the same effect as a box enclosing a whole text-like component.
    Overlapping OCR boxes are combined before counting to avoid double-counts.
    """
    height, width = component.shape
    covered = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x0 = max(0, int(np.floor(box[0] - bbox[0])))
        y0 = max(0, int(np.floor(box[1] - bbox[1])))
        x1 = min(width, int(np.ceil(box[2] - bbox[0])))
        y1 = min(height, int(np.ceil(box[3] - bbox[1])))
        if x0 < x1 and y0 < y1:
            covered[y0:y1, x0:x1] = True
    return (float(np.logical_and(component, covered).sum()) / max(1, int(component.sum())),
            float(covered.mean()))


def detect_direction_arrows(image, text_boxes=(), min_iou=0.82, min_margin=0.16,
                            min_side=16, min_area=120, max_text_overlap=0.25):
    """Find isolated bright arrows; output evidence in input-image coordinates.

    Several luminance thresholds accommodate dark/blue-tinted foreground. OCR
    overlap rejects the *whole* component when more than ``max_text_overlap``
    of its actual foreground pixels lie in the union of OCR boxes. Small edge
    intersections are allowed, but no strokes are erased and no fragments are
    classified. Components touching the crop edge are rejected. Connected road
    diagrams remain unsupported even if one arrow-shaped part is visible.
    The default requires at least three quarters of the component to lie
    outside every OCR span; it is a development heuristic, not a probability.
    """
    rgb = _load_rgb(image)
    height, width = rgb.shape[:2]
    boxes = [item.get("bbox") if isinstance(item, dict) else item for item in text_boxes]
    boxes = [list(box) for box in boxes if box is not None and len(box) == 4]
    channels = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), rgb[:, :, 0], rgb[:, :, 1])
    candidates, rejected = [], Counter()
    text_rejection_examples = []
    seen_masks = set()
    for channel_index, channel in enumerate(channels):
        otsu, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds = sorted(set([int(otsu)] + [int(np.percentile(channel, p)) for p in (55, 65, 75, 85)]))
        for threshold in thresholds:
            mask = (channel > threshold).astype(np.uint8)
            # Avoid evaluating identical flat-color masks repeatedly.
            identity = hashlib.sha256(mask.tobytes()).digest()
            if identity in seen_masks:
                continue
            seen_masks.add(identity)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, count):
                x, y, w, h, area = (int(value) for value in stats[label])
                if min(w, h) < min_side or area < min_area:
                    continue
                bbox = [x, y, x + w, y + h]
                if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                    rejected["touches_crop_edge"] += 1
                    continue
                if area > height * width * 0.18:
                    rejected["too_large"] += 1
                    continue
                component = labels[y:y + h, x:x + w] == label
                text_fraction, box_fraction = _text_coverage(component, bbox, boxes)
                if text_fraction > max_text_overlap:
                    rejected["overlaps_text"] += 1
                    if 0 < text_fraction < 0.65 and len(text_rejection_examples) < 12:
                        if not any(_overlap(bbox, row["bbox"]) > 0.8 for row in text_rejection_examples):
                            text_rejection_examples.append({"bbox": bbox,
                                "foreground_text_fraction": round(text_fraction, 4),
                                "bbox_text_fraction": round(box_fraction, 4)})
                    continue
                result = classify_arrow_mask(component, min_iou, min_margin)
                if result["direction"] is None:
                    rejected[result["reason"]] += 1
                    continue
                # High thresholds can sever a dim branch from a road diagram,
                # leaving an apparently standalone arrow. Check connectivity
                # in a more inclusive mask before accepting that fragment.
                border = channel[max(0, y - 3):min(height, y + h + 3),
                                 max(0, x - 3):min(width, x + w + 3)]
                background = float(np.percentile(border, 20))
                # A lower threshold below the local background floods the
                # whole sign and says nothing about branch connectivity.
                lower = max(int(background) + 1, int(threshold * 0.70))
                lo_mask = (channel > lower).astype(np.uint8)
                _, lo_labels, lo_stats, _ = cv2.connectedComponentsWithStats(lo_mask, connectivity=8)
                yy, xx = np.nonzero(component)
                connected_labels, counts = np.unique(lo_labels[y + yy, x + xx], return_counts=True)
                lo_label = int(connected_labels[np.argmax(counts)])
                lx, ly, lw, lh, la = (int(value) for value in lo_stats[lo_label])
                if (la > area * 1.15 and (lw > w * 1.4 or lh > h * 1.4)):
                    rejected["connected_structure_at_lower_threshold"] += 1
                    continue
                candidates.append({**result, "bbox": bbox,
                                   "center": [x + w / 2, y + h / 2], "area": area,
                                   "foreground_text_fraction": round(text_fraction, 4),
                                   "bbox_text_fraction": round(box_fraction, 4),
                                   "threshold_channel": ("gray", "red", "green")[channel_index],
                                   "threshold": threshold})
    kept = []
    for candidate in sorted(candidates, key=lambda row: row["shape_iou"], reverse=True):
        nearby = [other for other in candidates if _overlap(candidate["bbox"], other["bbox"]) > 0.4]
        if any(other["direction"] != candidate["direction"] for other in nearby):
            rejected["threshold_direction_conflict"] += 1
            continue
        if any(_overlap(candidate["bbox"], other["bbox"]) > 0.3 for other in kept):
            continue
        kept.append(candidate)
    kept.sort(key=lambda row: (row["bbox"][1], row["bbox"][0]))
    for index, arrow in enumerate(kept, 1):
        arrow["arrow_id"] = f"a{index}"
    return {"arrows": kept, "method": "standalone_shape_iou_v2",
            "text_boxes": len(boxes), "rejected": dict(rejected),
            "text_exclusion": {"method": "whole_component_foreground_coverage",
                               "max_foreground_fraction": max_text_overlap,
                               "retained_despite_bbox_intersection": sum(row["bbox_text_fraction"] > 0 for row in kept),
                               "rejection_examples": text_rejection_examples},
            "status": "ok" if kept else "abstain",
            "limitations": "cardinal standalone arrows only; geometric scores are not probabilities"}
