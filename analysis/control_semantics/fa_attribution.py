"""Attribute YOLOX-Tiny's residual zone false alarms (EXP-0010B preregistered v2 image-grid metric) to causes. POST-HOC, development data.

A false-alarm cell is a (frame, zone) that the v2 metric calls NEG and the detector marks occupied. Each such cell is
attributed to the detection that triggered it (highest score if several), classified by that detection's OWN box against the
visible boxes of every ground-truth class, with one fixed priority order (first match wins):

  1 humanoid            best IoU >= 0.3 with a FourierGR1T2/AgilityDigit box          (class ambiguity)
  2 truncated_detection box touches the frame edge (3 px)                              (frame-edge truncation)
  3 person_ignored      best IoU >= 0.3 with a Person that is ignored by the metric    (annotation scope: <32 px or foot outside frame / bad depth)
  4 person_boundary     best IoU >= 0.3 with a scored Person whose projected foot is within 2*margin of the cell (zone-boundary)
  5 person_foot_mismatch best IoU >= 0.3 with a scored Person farther than that        (label/foot-projection vs detector foot)
  6 person_partial      best Person IoU in [0.1, 0.3)                                  (localization / crowd)
  7 other_object        best IoU >= 0.3 with Forklift/PalletTruck/NovaCarter/Transporter (detector confusion on a non-person object)
  8 low_score_clutter   no overlap >= 0.1 with any GT object and score < 0.3
  9 unexplained         no overlap and score >= 0.3                                    (candidate detector error / unlabelled person -> crop audit)

Output: results/control_semantics/fa_attribution.json and, for the unexplained bucket, a crop montage under fp_audit/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import run_cag
from analysis.control_gate import zones as Z
from analysis.edge_inference import evaluate as ev

OUT = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
ROLES = (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1))
TAU = 0.05
EDGE = 3.0
CATS = ["humanoid", "truncated_detection", "person_ignored", "person_boundary", "person_foot_mismatch", "person_partial",
        "other_object", "low_score_clutter", "unexplained"]
HUM, OTHER = (4, 5), (1, 2, 3, 6)


def classify(det_box, score, frame_rows, g, clip, row_of, cell_zone_edges):
    """-> category for one detection (see module docstring)."""
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
    if x1 <= EDGE or y1 <= EDGE or x2 >= Z.W - EDGE or y2 >= Z.H - EDGE:
        return "truncated_detection"
    pi, pj = best(cls == 0)
    if pi >= 0.3:
        r = rows[pj]
        k = row_of.get(int(r))
        if k is None:
            return "person_foot_mismatch"
        fx, fy = clip.foot[k]
        if clip.vis_h[k] < Z.SIZE_FLOOR or not (0 <= fx < Z.W and 0 <= fy < Z.H) or not clip.depth_ok[k]:
            return "person_ignored"
        # distance from the person's foot to the nearest zone boundary region of the false-alarm cell is evaluated by the caller
        return "person_scored"
    if pi >= 0.1:
        return "person_partial"
    oi, _ = best(np.isin(cls, OTHER))
    if oi >= 0.3:
        return "other_object"
    if iou.size and iou.max() >= 0.1:
        return "other_object"
    return "low_score_clutter" if score < 0.3 else "unexplained"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"tau": TAU, "metric": "EXP-0010B preregistered v2 image-grid zone metric", "priority_order": CATS, "clips": {}}
    unexplained = []
    for role, stride in ROLES:
        clip, _ = run_cag.load_clip(role)
        g = clip.g
        dets, n = run_cag.load_dets(role, "YT", stride)
        frames = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
        pos, neg = Z.gt_states(clip.n_frames, clip.foot, clip.frame, clip.vis_h, clip.depth_ok)
        row_of = {int(r): k for k, r in enumerate(clip.prow)}
        rows_by_f = {}
        for i, fr in enumerate(g["frame"]):
            rows_by_f.setdefault(int(fr), []).append(i)
        keep = (dets["score"] >= TAU) & ((dets["box"][:, 3] - dets["box"][:, 1]) >= Z.SIZE_FLOOR)
        f, b, s = dets["frame"][keep], dets["box"][keep], dets["score"][keep]
        foot = np.stack([(b[:, 0] + b[:, 2]) / 2, np.minimum(b[:, 3], Z.H - 1e-3)], 1)
        zm = Z.zone_masks(foot, Z.MARGIN)
        fset = np.zeros(clip.n_frames, bool)
        fset[frames] = True
        cell_best = {}                                   # (frame, zone) -> (score, det index)
        for i in range(len(f)):
            if not fset[f[i]]:
                continue
            for z in np.nonzero(zm[i])[0]:
                if neg[f[i], z]:
                    key = (int(f[i]), int(z))
                    if key not in cell_best or s[i] > cell_best[key][0]:
                        cell_best[key] = (float(s[i]), i)
        n_neg = int(neg[frames].sum())
        counts = {c: 0 for c in CATS}
        counts["person_scored"] = 0
        sample_unexpl = []
        for (fr, z), (sc, i) in cell_best.items():
            cat = classify(b[i], sc, rows_by_f.get(fr, []), g, clip, row_of, None)
            if cat == "person_scored":
                # boundary vs foot mismatch: scored person's foot within 2*margin of this cell?
                gb = np.stack([[g["x1"][r], g["y1"][r], g["x2"][r], g["y2"][r]] for r in rows_by_f.get(fr, [])])
                cls = np.asarray([g["cls"][r] for r in rows_by_f.get(fr, [])])
                iou = ev.box_iou(b[i][None], gb)[0]
                j = int(np.argmax(np.where(cls == 0, iou, -1)))
                k = row_of[int(rows_by_f[fr][j])]
                near = Z.zone_masks(clip.foot[k][None], 2 * Z.MARGIN)[0, z]
                cat = "person_boundary" if near else "person_foot_mismatch"
            counts[cat] += 1
            if cat == "unexplained" and len(sample_unexpl) < 4000:
                sample_unexpl.append((role, fr, b[i].tolist(), sc))
        counts.pop("person_scored")
        n_fa = len(cell_best)
        out["clips"][role] = {"stride": stride, "frames_scored": int(len(frames)), "neg_cells": n_neg, "false_alarm_cells": n_fa,
                              "fpr_v2": round(n_fa / max(n_neg, 1), 4),
                              "cause_cells": counts, "cause_fraction_of_fa": {c: round(v / max(n_fa, 1), 4) for c, v in counts.items()},
                              "cause_contribution_to_fpr": {c: round(v / max(n_neg, 1), 4) for c, v in counts.items()}}
        print(role, "FPR", round(n_fa / max(n_neg, 1), 3), {c: round(v / max(n_fa, 1), 3) for c, v in counts.items()}, flush=True)
        unexplained += sample_unexpl
    (OUT / "fa_attribution.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "fa_unexplained_sample.json").write_text(json.dumps(unexplained[:4000], default=float))


if __name__ == "__main__":
    sys.exit(main())
