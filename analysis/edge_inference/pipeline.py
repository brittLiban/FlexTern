"""Shared preprocessing and postprocessing for SSDLite320-MobileNetV3-Large (person only).

Pure numpy + OpenCV so the *same* code runs in front of the PyTorch (B0) and the ONNX
Runtime (B1) network. The network is the only component that differs between runtimes.

Semantics follow torchvision 0.26 SSD (verified against `model(img)` in parity checks):
  - normalisation mean 0.5 / std 0.5 (i.e. x/127.5 - 1), non-aspect-preserving resize to 320x320
  - softmax over 91 COCO logits, label 1 (person), score > 0.001, top-300
  - BoxCoder weights (10, 10, 5, 5), xform clip log(1000/16), clip to the 320x320 image
  - NMS IoU 0.55, boxes rescaled to the native frame size
The one documented deviation is that the official 300-detection cap is applied across all
classes, whereas this person-only path caps person detections only.
"""
from __future__ import annotations

import math

import numpy as np

INPUT_SIZE = 320
PERSON_LABEL = 1
SCORE_FLOOR = 0.001
TOPK = 300
NMS_IOU = 0.55
MAX_DETS = 300
BBOX_XFORM_CLIP = math.log(1000.0 / 16)
WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB -> float32 1x3x320x320 normalised to [-1, 1]."""
    import cv2
    small = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = small.astype(np.float32)
    x *= np.float32(1.0 / 127.5)
    x -= np.float32(1.0)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def decode_boxes(rel: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Numpy port of torchvision BoxCoder.decode_single (xyxy anchors, relative codes)."""
    wx, wy, ww, wh = WEIGHTS
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * widths
    cy = anchors[:, 1] + 0.5 * heights
    dx = rel[:, 0] / wx
    dy = rel[:, 1] / wy
    dw = np.minimum(rel[:, 2] / ww, BBOX_XFORM_CLIP)
    dh = np.minimum(rel[:, 3] / wh, BBOX_XFORM_CLIP)
    pcx = dx * widths + cx
    pcy = dy * heights + cy
    pw = np.exp(dw) * widths
    ph = np.exp(dh) * heights
    out = np.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], axis=1)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score (suppress IoU > iou_thr)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for ii in range(len(order)):
        if suppressed[ii]:
            continue
        i = order[ii]
        keep.append(i)
        rest = order[ii + 1:]
        if len(rest) == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter)
        suppressed[ii + 1:] |= iou > iou_thr
    return np.asarray(keep, dtype=np.int64)


def softmax_person(cls_logits: np.ndarray) -> np.ndarray:
    z = cls_logits - cls_logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, PERSON_LABEL] / e.sum(axis=1)


def postprocess(cls_logits: np.ndarray, bbox_reg: np.ndarray, anchors: np.ndarray,
                frame_w: int, frame_h: int):
    """Head outputs ([A,91], [A,4]) -> (boxes[N,4] native-pixel xyxy float32, scores[N]) sorted by score."""
    scores = softmax_person(cls_logits)
    idx = np.nonzero(scores > SCORE_FLOOR)[0]
    if len(idx) > TOPK:
        idx = idx[np.argsort(-scores[idx], kind="stable")[:TOPK]]
    if len(idx) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    boxes = decode_boxes(bbox_reg[idx].astype(np.float32), anchors[idx])
    np.clip(boxes, 0, INPUT_SIZE, out=boxes)
    sc = scores[idx]
    keep = nms(boxes, sc, NMS_IOU)[:MAX_DETS]
    boxes = boxes[keep] * np.array([frame_w / INPUT_SIZE, frame_h / INPUT_SIZE] * 2, dtype=np.float32)
    return boxes.astype(np.float32), sc[keep].astype(np.float32)
