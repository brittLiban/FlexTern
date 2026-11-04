"""POST-HOC false-alarm attribution v2 (EXP-0010C, after the advisor's completion review). Writes NEW files; fa_attribution.json,
fa_attribution_gp.json and their code are untouched (the first is hashed in lock.json).

Why v2: in fa_attribution.classify the humanoid test needs IoU >= 0.3 and the person-partial test looks at Person boxes only, so a detection
overlapping a humanoid at IoU 0.1-0.3, or a vehicle at 0.1-0.3, fell through to the generic 'other_object' bucket. v2 splits it:

  humanoid                   IoU >= 0.3 with a humanoid box (class ambiguity)          [v2 metric: position-agnostic; ground metric: humanoid_position_mismatch]
  truncated_detection        box touches the frame edge
  person_ignored             IoU >= 0.3 with an ignored Person
  person_boundary / person_foot_mismatch (image grid) or person_position_mismatch (ground): IoU >= 0.3 with a scored Person
  person_partial             Person IoU in [0.1, 0.3)
  humanoid_partial           humanoid IoU in [0.1, 0.3)
  vehicle_or_amr             IoU >= 0.3 with Forklift / PalletTruck / NovaCarter / Transporter   (detector confusion on a non-person object)
  vehicle_or_amr_partial     those classes at IoU in [0.1, 0.3)
  low_score_clutter          no overlap >= 0.1 with any GT box, score < 0.3
  unexplained                no overlap, score >= 0.3

Outputs: results/control_semantics/fa_attribution_v2.json (image-grid metric, development clips), fa_attribution_gp_v2.json (policy B, six clips,
with the cell-level false alarms left if the two position-mismatch buckets were label noise) and crop montages for hold_L03.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from analysis.control_gate import zones as Z
from analysis.edge_inference import evaluate as ev
from . import fa_attribution as FA
from . import fa_attribution_gp as FG
from . import gp_metric as GP
from . import ground, run_cs

RES = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
HUM, VEH = (4, 5), (1, 2, 3, 6)


def classify2(det_box, score, frame_rows, g, clip, row_of, cell_zone_edges=None):
    rows = frame_rows
    if len(rows):
        gb = np.stack([[g["x1"][r], g["y1"][r], g["x2"][r], g["y2"][r]] for r in rows])
        iou = ev.box_iou(det_box[None], gb)[0]
        cls = np.asarray([g["cls"][r] for r in rows])
    else:
        iou, cls = np.zeros(0), np.zeros(0, int)

    def best(mask):
        if not mask.any():
            return 0.0, -1
        j = int(np.argmax(np.where(mask, iou, -1)))
        return float(iou[j]), j

    hi, _ = best(np.isin(cls, HUM))
    if hi >= 0.3:
        return "humanoid"
    x1, y1, x2, y2 = det_box
    if x1 <= FA.EDGE or y1 <= FA.EDGE or x2 >= Z.W - FA.EDGE or y2 >= Z.H - FA.EDGE:
        return "truncated_detection"
    pi, pj = best(cls == 0)
    if pi >= 0.3:
        k = row_of.get(int(rows[pj]))
        if k is None:
            return "person_scored"
        fx, fy = clip.foot[k]
        if clip.vis_h[k] < Z.SIZE_FLOOR or not (0 <= fx < Z.W and 0 <= fy < Z.H) or not clip.depth_ok[k]:
            return "person_ignored"
        return "person_scored"
    if pi >= 0.1:
        return "person_partial"
    if hi >= 0.1:
        return "humanoid_partial"
    vi, _ = best(np.isin(cls, VEH))
    if vi >= 0.3:
        return "vehicle_or_amr"
    if vi >= 0.1:
        return "vehicle_or_amr_partial"
    return "low_score_clutter" if score < 0.3 else "unexplained"


def attribute_ground(role, stride, tau, samples=None):
    """FG.attribute with the v2 classifier (monkeypatched for the call), optionally collecting per-bucket samples."""
    old = FA.classify
    log = []

    def wrapped(det_box, score, frame_rows, g, clip, row_of, z):
        c = classify2(det_box, score, frame_rows, g, clip, row_of, z)
        log.append((c, det_box.copy(), float(score), frame_rows))
        return c
    FA.classify = wrapped
    try:
        res = FG.attribute(role, stride, tau)
    finally:
        FA.classify = old
    return res, log


CATS2 = ["humanoid", "truncated_detection", "person_ignored", "person_boundary", "person_foot_mismatch", "person_partial", "humanoid_partial",
         "vehicle_or_amr", "vehicle_or_amr_partial", "low_score_clutter", "unexplained"]


def image_grid_v2():
    """Re-run the EXP-0010B image-grid attribution with the v2 classifier, redirecting its output so fa_attribution.json is not overwritten."""
    tmp = RES / "_fa_v2_tmp"
    tmp.mkdir(exist_ok=True)
    old = (FA.classify, FA.OUT, FA.CATS)
    FA.classify, FA.OUT, FA.CATS = classify2, tmp, CATS2
    try:
        FA.main()
    finally:
        FA.classify, FA.OUT, FA.CATS = old
    (RES / "fa_attribution_v2.json").write_text((tmp / "fa_attribution.json").read_text())
    (RES / "fa_unexplained_sample_v2.json").write_text((tmp / "fa_unexplained_sample.json").read_text())
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def main():
    run_cag.CLIPS.update(run_cs.fresh_clips())
    image_grid_v2()
    lock = json.loads(run_cs.LOCK.read_text())
    tau = lock["tau"]["B_person_shaped_agent"]
    out = {"note": "POST-HOC v2 classifier; policy B, locked tau; run after the fresh scoring", "clips": []}
    montage_samples = {}
    for role, stride in (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1), ("hold_L05", 1), ("hold_L03", 1)):
        res, log = attribute_ground(role, stride, tau)
        mm = res["cause_cells"].get("person_position_mismatch", 0) + res["cause_cells"].get("humanoid_position_mismatch", 0)
        n_fa, pred_cells = res["false_alarm_cells"], res["predicted_cells_scored"]
        res["fa_per_predicted_if_position_mismatch_buckets_were_label_noise"] = (n_fa - mm) / max(pred_cells, 1)
        out["clips"].append(res)
        print(role, "FA/pred %.3f" % res["fa_per_predicted"], "if mismatch buckets were noise: %.3f" % res["fa_per_predicted_if_position_mismatch_buckets_were_label_noise"],
              res["cause_fraction_of_fa"], flush=True)
        if role == "hold_L03":
            montage_samples = log
    (RES / "fa_attribution_gp_v2.json").write_text(json.dumps(out, indent=1, default=float))
    make_montages(montage_samples)


def make_montages(log):
    """One crop montage per bucket on hold_L03 (the clip that decides the verdict). Needs cv2 + av (WSL venv)."""
    import cv2
    from analysis.control_gate import parity as P
    clip, video = run_cag.load_clip("hold_L03")
    g = clip.g
    # the log holds (category, box, score, rows) but not the frame; rebuild the frame list from the rows' frame
    picks = {}
    want = {"vehicle_or_amr": 12, "low_score_clutter": 12, "humanoid": 12, "person_scored": 12}
    for c, box, sc, rows in log:
        if c in want and len(rows):
            picks.setdefault(c, []).append((int(g["frame"][rows[0]]), box, sc, rows))
    sel = {}
    for c, items in picks.items():
        idx = np.linspace(0, len(items) - 1, min(want[c], len(items))).astype(int)
        sel[c] = [items[i] for i in idx]
    frames = sorted({f for v in sel.values() for f, *_ in v})
    imgs = P._frames(video, frames)
    out = RES / "fp_audit"
    out.mkdir(exist_ok=True)
    for c, items in sel.items():
        th = []
        for f, box, sc, rows in items:
            img = imgs[f].copy()
            for r in rows:
                k = g["cls"][r]
                col = (0, 255, 0) if k == 0 else (255, 0, 255) if k in (4, 5) else (255, 140, 0)
                cv2.rectangle(img, (int(g["x1"][r]), int(g["y1"][r])), (int(g["x2"][r]), int(g["y2"][r])), col, 3)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 4)
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            hf = int(max(box[2] - box[0], box[3] - box[1]) * 0.9) + 50
            crop = img[max(cy - hf, 0):min(cy + hf, 1080), max(cx - hf, 0):min(cx + hf, 1920)]
            crop = cv2.resize(crop, (256, 256))
            cv2.putText(crop, f"f{f} s{sc:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            th.append(crop)
        while len(th) % 4:
            th.append(np.zeros_like(th[0]))
        mont = np.concatenate([np.concatenate(th[i:i + 4], 1) for i in range(0, len(th), 4)], 0)
        name = f"hold_L03_B_{c}.jpg"
        cv2.imwrite(str(out / name), mont[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("montage", name, len(items))
    (out / "hold_L03_B_montage_legend.json").write_text(json.dumps({"red": "detection", "green": "Person GT", "magenta": "humanoid GT",
                                                                    "orange": "Forklift/PalletTruck/NovaCarter/Transporter GT"}))


if __name__ == "__main__":
    sys.exit(main())
