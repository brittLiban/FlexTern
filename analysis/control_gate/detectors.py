"""Candidate detector configurations for the control-adequacy gate (all run through ONNX Runtime CPU, torch-free at run time).

S0/S1/S2 : torchvision SSDLite320-MobileNetV3-Large COCO weights at 320x320 (Stage A), 576x320 and 416x240.
YT/YS    : YOLOX-Tiny (416) / YOLOX-S (640), official ONNX files of release 0.1.1rc0.
Every config exposes the harness `Engine` (pre -> infer -> post) so the Stage A replay/offline loops are reused unchanged.
Output of post(): (boxes[N,4] native-pixel xyxy float32, scores[N]) sorted by descending score, person only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.edge_inference import models as sa_models
from analysis.edge_inference import pipeline as sa_pipe
from analysis.edge_inference.common import PKG, cache_dir
from analysis.edge_inference.replay import Engine

SSD_SIZES = {"S0": (320, 320), "S1": (576, 320), "S2": (416, 240)}   # (W, H)
YOLOX = {"YT": ("yolox_tiny.onnx", 416), "YS": ("yolox_s.onnx", 640)}
SCORE_FLOOR = 0.01
SSD_NMS, YOLOX_NMS = 0.55, 0.45
TOPK = 300
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CONFIGS = ("S0", "S1", "S2", "YT", "YS")


# ---------------------------------------------------------------- SSDLite variants
def ssd_paths(name: str) -> tuple[Path, Path]:
    if name == "S0":
        return sa_models.onnx_cache_path(), sa_models.ANCHORS_PATH
    w, h = SSD_SIZES[name]
    d = cache_dir() / "models"
    return d / f"ssdlite320_mobilenet_v3_large_{w}x{h}.onnx", d / f"ssdlite320_mobilenet_v3_large_{w}x{h}_anchors.npy"


def export_ssd_variant(name: str):
    """Needs torch (WSL). Exports the network at (H, W) and the anchors the torchvision generator produces there."""
    import torch
    from torchvision.models.detection.image_list import ImageList
    w, h = SSD_SIZES[name]
    model, _ = sa_models.load_official()
    onnx_path, anchor_path = ssd_paths(name)
    net = sa_models._network_module(model)
    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        feats = model.backbone(x)
        feats = list(feats.values()) if not isinstance(feats, torch.Tensor) else [feats]
        anchors = model.anchor_generator(ImageList(x, [(h, w)]), feats)[0].numpy().astype(np.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, (x,), str(onnx_path), input_names=["input"], output_names=["cls_logits", "bbox_regression"],
                      opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    np.save(anchor_path, anchors)
    return onnx_path, anchor_path, anchors.shape


def ssd_pre(w: int, h: int):
    import cv2

    def pre(rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x *= np.float32(1.0 / 127.5)
        x -= np.float32(1.0)
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    return pre


def ssd_post(anchors: np.ndarray, w: int, h: int, floor: float = SCORE_FLOOR):
    def post(out, fw: int, fh: int):
        cls, reg = out
        scores = sa_pipe.softmax_person(cls)
        idx = np.nonzero(scores > floor)[0]
        if len(idx) > TOPK:
            idx = idx[np.argsort(-scores[idx], kind="stable")[:TOPK]]
        if len(idx) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes = sa_pipe.decode_boxes(reg[idx].astype(np.float32), anchors[idx])
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        sc = scores[idx]
        keep = sa_pipe.nms(boxes, sc, SSD_NMS)[:TOPK]
        boxes = boxes[keep] * np.array([fw / w, fh / h, fw / w, fh / h], dtype=np.float32)
        return boxes.astype(np.float32), sc[keep].astype(np.float32)
    return post


# ---------------------------------------------------------------- YOLOX
def yolox_path(name: str) -> Path:
    return cache_dir() / "models" / "yolox" / YOLOX[name][0]


def yolox_pre(size: int):
    """BGR, raw 0-255, letterbox top-left with pad 114, NO mean/std. (The release-0.1.1rc0 demo script normalises with ImageNet
    mean/std, but these ONNX weights then output person scores ~1e-3 on frames where raw 0-255 input gives 0.7-0.8; see
    amendment 1 in protocol_v2_amendment1.json.)"""
    import cv2

    def pre(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        r = min(size / h, size / w)
        resized = cv2.resize(rgb, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        pad = np.full((size, size, 3), 114.0, np.float32)
        pad[:resized.shape[0], :resized.shape[1]] = resized
        return np.ascontiguousarray(pad[:, :, ::-1].transpose(2, 0, 1)[None])
    return pre


def yolox_grid(size: int):
    grids, strides = [], []
    for s in (8, 16, 32):
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)


def yolox_post(size: int, floor: float = SCORE_FLOOR):
    grid, stride = yolox_grid(size)

    def post(out, fw: int, fh: int):
        p = out[0] if out.ndim == 3 else out
        sc = p[:, 4] * p[:, 5]                       # objectness * P(person) (COCO class 0)
        idx = np.nonzero(sc > floor)[0]
        if len(idx) > TOPK:
            idx = idx[np.argsort(-sc[idx], kind="stable")[:TOPK]]
        if len(idx) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        xy = (p[idx, :2] + grid[idx]) * stride[idx]
        wh = np.exp(p[idx, 2:4]) * stride[idx]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
        r = min(size / fh, size / fw)
        boxes = boxes / r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh)
        s = sc[idx]
        keep = sa_pipe.nms(boxes, s, YOLOX_NMS)[:TOPK]
        return boxes[keep].astype(np.float32), s[keep].astype(np.float32)
    return post


# ---------------------------------------------------------------- runtime
class OrtGeneric:
    """ONNX Runtime CPU session returning the raw network outputs (batch dim removed)."""

    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.providers = self.sess.get_providers()
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray):
        outs = self.sess.run(None, {self.in_name: x})
        return tuple(o[0] for o in outs) if len(outs) > 1 else outs[0]


def build_engine(name: str, threads: int) -> tuple[Engine, OrtGeneric]:
    if name in SSD_SIZES:
        w, h = SSD_SIZES[name]
        onnx_path, anchor_path = ssd_paths(name)
        runner = OrtGeneric(onnx_path, threads)
        anchors = np.load(anchor_path)
        return Engine(name, ssd_pre(w, h), runner, ssd_post(anchors, w, h)), runner
    size = YOLOX[name][1]
    runner = OrtGeneric(yolox_path(name), threads)
    return Engine(name, yolox_pre(size), runner, yolox_post(size)), runner


def model_files(name: str) -> list[Path]:
    if name in SSD_SIZES:
        a, b = ssd_paths(name)
        return [a, b]
    return [yolox_path(name)]
