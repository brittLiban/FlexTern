"""SSDLite320-MobileNetV3-Large: official weights, ONNX export, and the two network runners.

Only the *network* (backbone + SSDLite head, fixed 1x3x320x320 input) differs between
B0 (PyTorch FP32 eager) and B1 (ONNX Runtime CPU). Pre/postprocessing is shared
(`pipeline.py`). Anchors are a constant of the architecture at this input size and are
computed once from the official model and cached as an .npy beside this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import PKG, cache_dir, sha256_file

ANCHORS_PATH = PKG / "ssdlite320_anchors.npy"
N_ANCHORS = 3234
N_CLASSES = 91


def load_official(threads: int | None = None):
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    torch.manual_seed(0)
    if threads:
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set / parallel work already started
    w = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    m = ssdlite320_mobilenet_v3_large(weights=w).eval()
    return m, w


def checkpoint_path() -> Path | None:
    """Locate the cached official checkpoint (torch hub cache) for hashing.

    Deliberately does NOT import torch: B1 (ONNX Runtime) runs must stay torch-free so their
    memory footprint is the deployment footprint. Mirrors torch.hub.get_dir():
    $TORCH_HOME/hub, else $XDG_CACHE_HOME/torch/hub, else ~/.cache/torch/hub.
    """
    import os
    home = os.environ.get("TORCH_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), "torch")
    d = Path(home) / "hub" / "checkpoints"
    hits = sorted(d.glob("ssdlite320_mobilenet_v3_large_coco-*.pth"))
    return hits[0] if hits else None


def _network_module(model):
    import torch

    class Net(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone, self.head = m.backbone, m.head

        def forward(self, x):
            feats = self.backbone(x)
            feats = list(feats.values()) if not isinstance(feats, torch.Tensor) else [feats]
            out = self.head(feats)
            return out["cls_logits"], out["bbox_regression"]

    return Net(model).eval()


def compute_anchors(model) -> np.ndarray:
    import torch
    from torchvision.models.detection.image_list import ImageList
    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        feats = model.backbone(x)
        feats = list(feats.values()) if not isinstance(feats, torch.Tensor) else [feats]
        anchors = model.anchor_generator(ImageList(x, [(320, 320)]), feats)[0]
    return anchors.numpy().astype(np.float32)


def get_anchors(model=None) -> np.ndarray:
    if ANCHORS_PATH.exists():
        return np.load(ANCHORS_PATH)
    if model is None:
        model, _ = load_official()
    a = compute_anchors(model)
    np.save(ANCHORS_PATH, a)
    return a


def export_onnx(model, path: Path) -> Path:
    import torch
    net = _network_module(model)
    dummy = torch.zeros(1, 3, 320, 320)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, (dummy,), str(path), input_names=["input"], output_names=["cls_logits", "bbox_regression"],
                      opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path


class TorchRunner:
    """B0: PyTorch FP32 eager CPU network."""
    name = "B0_torch_fp32"

    def __init__(self, model, threads: int):
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        self.net = _network_module(model)

    def __call__(self, x: np.ndarray):
        torch = self.torch
        with torch.inference_mode():
            c, b = self.net(torch.from_numpy(x))
        return c[0].numpy(), b[0].numpy()


class OrtRunner:
    """B1: ONNX Runtime CPUExecutionProvider, ORT_ENABLE_ALL."""
    name = "B1_ort_cpu"

    def __init__(self, onnx_path: Path, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.providers = self.sess.get_providers()
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray):
        c, b = self.sess.run(None, {self.in_name: x})
        return c[0], b[0]


def onnx_cache_path() -> Path:
    return cache_dir() / "models" / "ssdlite320_mobilenet_v3_large_fp32.onnx"


def model_identity(onnx_path: Path | None = None) -> dict:
    ck = checkpoint_path()
    ident = {"checkpoint_file": ck.name if ck else None, "checkpoint_sha256": sha256_file(ck) if ck else None,
             "anchors_sha256": sha256_file(ANCHORS_PATH) if ANCHORS_PATH.exists() else None}
    p = onnx_path or onnx_cache_path()
    if p.exists():
        ident["onnx_file"] = p.name
        ident["onnx_sha256"] = sha256_file(p)
    return ident
