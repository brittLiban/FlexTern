"""Post-hoc false-positive diagnostic for EXP-0010 Stage A (no new inference; uses cached detections).

Question: is the low precision/recall mostly a *localisation / box-semantics* mismatch (the dataset's
GT boxes are the VISIBLE extent of each object, a COCO detector predicts the whole person), or does the
detector fire on nothing / on the wrong objects?

1. AP50-style AP and the operating point re-scored at IoU 0.1 / 0.3 / 0.5.
2. False positives at tau split by their best overlap with ANY ground-truth box of the same frame:
     near_miss_person : best IoU with a Person GT in [0.1, 0.5)   (extent / localisation mismatch)
     on_nonperson     : best IoU >= 0.1 with a non-Person GT (forklift, pallet truck, robot, ...)
     duplicate_hit    : IoU >= 0.5 with a Person GT that was already matched (duplicate)
     background       : best IoU < 0.1 with every GT box
The development and calibration clips are the primary evidence. The holdout is included ONLY as a labelled
post-hoc diagnostic; the frozen primary metric is not changed by anything here.

Usage: python -m analysis.edge_inference.diagnose_fp
"""
from __future__ import annotations

import json

import numpy as np

from . import evaluate as ev
from . import gt as gtmod
from .common import RESULTS, cache_dir, load_protocol
from .run_stage_a import clip_name, clip_paths


def _frame_arrays(gt, n):
    order = np.argsort(gt["frame"], kind="stable")
    fs = gt["frame"][order]
    starts = np.searchsorted(fs, np.arange(n + 1))
    box = np.stack([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], 1)[order].astype(np.float64)
    return box, gt["cls"][order], starts


def split_false_positives(cs: ev.ClipScore, dets: dict, gt: dict, tau: float, iou_thr: float = 0.5) -> dict:
    n = gt["n_frames"]
    gbox, gcls, starts = _frame_arrays(gt, n)
    counts = {"near_miss_person": 0, "on_nonperson": 0, "duplicate_hit": 0, "background": 0}
    sel = (cs.det_flag == 0) & (cs.det_score >= tau)
    fr = cs.det_frame[sel]
    # recover the detection boxes in the same order as cs (frame-grouped, score-descending)
    dmap = ev._group(dets["frame"], n)
    boxes_by_frame = {}
    for f in np.unique(fr):
        di = dmap[int(f)]
        o = np.argsort(-dets["score"][di], kind="stable")
        boxes_by_frame[int(f)] = (dets["box"][di][o], dets["score"][di][o])
    for f in np.unique(fr):
        b, s = boxes_by_frame[int(f)]
        keep = s >= tau
        b, s = b[keep], s[keep]
        m_f = (cs.det_frame == f) & (cs.det_score >= tau)
        flags = cs.det_flag[m_f]
        fp_idx = np.nonzero(flags == 0)[0]
        g0, g1 = starts[f], starts[f + 1]
        if g1 == g0:
            counts["background"] += len(fp_idx)
            continue
        iou = ev.box_iou(b[fp_idx], gbox[g0:g1])
        is_person = gcls[g0:g1] == 0
        for row in iou:
            best_p = row[is_person].max() if is_person.any() else 0.0
            best_n = row[~is_person].max() if (~is_person).any() else 0.0
            if best_p >= iou_thr:
                counts["duplicate_hit"] += 1
            elif best_p >= 0.1 and best_p >= best_n:
                counts["near_miss_person"] += 1
            elif best_n >= 0.1:
                counts["on_nonperson"] += 1
            elif best_p >= 0.1:
                counts["near_miss_person"] += 1
            else:
                counts["background"] += 1
    tot = sum(counts.values())
    return {"fp_total_checked": tot, **counts, **{f"frac_{k}": (v / tot if tot else float("nan")) for k, v in counts.items()}}


def main():
    tau = json.loads((RESULTS / "calibration.json").read_text())["tau"]
    proto = load_protocol()
    out = {"tau": tau, "note": "post-hoc diagnostic; primary frozen metric is IoU 0.5 with the 32 px floor", "clips": {}}
    for role in ("calibration", "development", "holdout"):
        g = gtmod.load_camera_gt(clip_paths(role)[1])
        d = np.load(cache_dir() / "runs" / clip_name(role) / "B0" / "offline_r0_dets.npz")
        dets = {k: d[k] for k in d.files}
        n = g["n_frames"]
        res = {"label": "post-hoc diagnostic (holdout)" if role == "holdout" else "diagnostic", "by_iou": {}}
        for thr in (0.1, 0.3, 0.5):
            cs = ev.score_clip(n, dets, g, iou_thr=thr)
            m = ev.clip_metrics(cs, tau)
            res["by_iou"][str(thr)] = {k: m[k] for k in ("ap50", "precision", "recall", "f1", "tp", "fp", "fn")}
            if thr == 0.5:
                res["fp_split_at_tau"] = split_false_positives(cs, dets, g, tau)
        out["clips"][role] = res
        print(role, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if kk in ("ap50", "precision", "recall", "f1")} for k, v in res["by_iou"].items()}))
        print("   FP split:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["fp_split_at_tau"].items()})
    (RESULTS / "fp_diagnostic.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float), encoding="utf8")


if __name__ == "__main__":
    main()
