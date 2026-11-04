"""Export the SSDLite variants and verify every candidate's pipeline against an independent reference *before* any scoring.

SSD variants : (a) raw heads torch vs ORT, (b) person detections of the numpy path vs the official torchvision model
               run with `transform.fixed_size` set to the variant (W, H).
YOLOX        : numpy port vs the release's own demo_postprocess/multiclass_nms (file fetched from the 0.1.1rc0 tag,
               read before use, hash recorded), and the preprocessing vs that tag's data_augment.preproc.
Writes results/control_gate/parity.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from analysis.edge_inference.common import cache_dir, sha256_file
from analysis.edge_inference.replay import _open, count_packets
from . import detectors as D

OUT = Path(__file__).resolve().parents[2] / "results" / "control_gate"


def _frames(video: Path, idxs: list[int]):
    data = video.read_bytes()
    c, s = _open(data)
    want, out = set(idxs), {}
    for i, fr in enumerate(c.decode(s)):
        if i in want:
            out[i] = fr.to_ndarray(format="rgb24")
        if i >= max(idxs):
            break
    c.close()
    return out


def parity_ssd(name: str, video: Path, idxs: list[int]) -> dict:
    import torch
    from analysis.edge_inference import models as sa_models
    w, h = D.SSD_SIZES[name]
    model, _ = sa_models.load_official(threads=4)
    model.transform.fixed_size = (w, h)   # torchvision interprets fixed_size as (width, height)
    net = sa_models._network_module(model)
    eng, runner = D.build_engine(name, 4)
    frames = _frames(video, idxs)
    head_diff_c = head_diff_b = 0.0
    same_count, dev_box = 0, 0.0
    for i, rgb in frames.items():
        x = eng.pre(rgb)
        c_o, b_o = eng.infer(x)
        with torch.inference_mode():
            c_t, b_t = net(torch.from_numpy(x))
        head_diff_c = max(head_diff_c, float(np.abs(c_t[0].numpy() - c_o).max()))
        head_diff_b = max(head_diff_b, float(np.abs(b_t[0].numpy() - b_o).max()))
        boxes, scores = eng.post((c_o, b_o), rgb.shape[1], rgb.shape[0])
        with torch.inference_mode():
            off = model([torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0])[0]
        m = (off["labels"] == 1) & (off["scores"] >= 0.3)
        ob, osc = off["boxes"][m].numpy(), off["scores"][m].numpy()
        mine = scores >= 0.3
        same_count += int(mine.sum() == len(osc))
        if mine.sum() == len(osc) and len(osc):
            a = boxes[mine][np.argsort(-scores[mine])]
            b = ob[np.argsort(-osc)]
            dev_box = max(dev_box, float(np.abs(a - b).max()))
    return {"config": name, "frames": len(frames), "max_abs_cls_logit_diff_torch_vs_ort": head_diff_c,
            "max_abs_bbox_diff_torch_vs_ort": head_diff_b, "frames_equal_count_at_0.3_vs_official": same_count,
            "max_box_dev_px_vs_official_when_counts_equal": dev_box,
            "onnx_sha256": sha256_file(D.ssd_paths(name)[0]), "anchors_sha256": sha256_file(D.ssd_paths(name)[1]),
            "note": "official path: torchvision SSD with transform.fixed_size=(W,H) (torchvision convention), float torch resize; mine: OpenCV uint8 resize (Stage A showed ~8 px at 1080p from the resize implementation alone)"}


def _load_ref(path: Path):
    spec = importlib.util.spec_from_file_location("yolox_demo_utils_ref", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parity_yolox(name: str, video: Path, idxs: list[int]) -> dict:
    import cv2
    ref_path = cache_dir() / "models" / "yolox" / "demo_utils_ref.py"
    ref = _load_ref(ref_path)
    size = D.YOLOX[name][1]
    eng, runner = D.build_engine(name, 4)
    frames = _frames(video, idxs)
    same, pre_dev, box_dev, nonempty = 0, 0.0, 0.0, 0
    for i, rgb in frames.items():
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        # reference input: BGR raw 0-255 letterbox (YOLOX training convention; amendment 1), built independently of detectors.py
        padded = np.ones((size, size, 3), np.float32) * 114.0
        r = min(size / bgr.shape[0], size / bgr.shape[1])
        resized = cv2.resize(bgr, (int(bgr.shape[1] * r), int(bgr.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        padded[: int(bgr.shape[0] * r), : int(bgr.shape[1] * r)] = resized
        ref_in = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None]
        mine_in = eng.pre(rgb)
        pre_dev = max(pre_dev, float(np.abs(ref_in - mine_in).max()))
        out_ref = runner.sess.run(None, {runner.in_name: ref_in})[0]
        pred = ref.demo_postprocess(out_ref.copy(), (size, size))[0]
        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]
        xyxy = np.ones_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        xyxy /= r
        dets = ref.multiclass_nms(xyxy, scores, nms_thr=0.45, score_thr=0.3)
        pr = dets[dets[:, 5] == 0] if dets is not None else np.zeros((0, 6))
        mb, ms = eng.post(runner(mine_in), rgb.shape[1], rgb.shape[0])
        k = ms >= 0.3
        nonempty += int(len(pr) > 0)
        if k.sum() == len(pr):
            same += 1
            if len(pr):
                a = mb[k][np.argsort(-ms[k])]
                b = pr[np.argsort(-pr[:, 4])][:, :4]
                box_dev = max(box_dev, float(np.abs(a - b).max()))
    return {"config": name, "frames": len(frames), "max_abs_preprocess_diff_vs_release_preproc": pre_dev,
            "frames_equal_person_count_at_0.3_vs_release_postprocess": same, "frames_with_any_person_at_0.3_reference": nonempty, "max_box_dev_px_when_counts_equal": box_dev,
            "onnx_sha256": sha256_file(D.yolox_path(name)), "reference_demo_utils_sha256": sha256_file(ref_path),
            "note": "reference NMS uses the +1 pixel area convention; mine does not (sub-pixel effect)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    video = cache_dir() / "raw" / "train" / "Warehouse_017" / "videos" / "Camera_0001.mp4"
    idxs = list(range(0, 9000, 225))     # 40 frames
    res = {"export": {}, "parity": []}
    for n in ("S1", "S2"):
        if not D.ssd_paths(n)[0].exists():
            res["export"][n] = [str(x) for x in D.export_ssd_variant(n)[2:]]
    for n in D.SSD_SIZES:
        res["parity"].append(parity_ssd(n, video, idxs))
        print(json.dumps(res["parity"][-1]), flush=True)
    for n in D.YOLOX:
        if D.yolox_path(n).exists() and D.yolox_path(n).stat().st_size >= {"YT": 20219662, "YS": 35858002}[n]:   # skip incomplete downloads
            res["parity"].append(parity_yolox(n, video, idxs))
            print(json.dumps(res["parity"][-1]), flush=True)
    (OUT / "parity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
