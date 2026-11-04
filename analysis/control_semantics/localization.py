"""Measured foot localization of YOLOX-Tiny detections, in pixels and in ground-plane metres, on development clips.

For every scored, interior Person (visible height >= 32 px, box wholly inside the frame) matched to a detection of score >= TAU
with IoU >= 0.5 against its visible box, compare the detection's foot point (box bottom-centre) with the ground truth:
  - image space: detection foot minus the 3D-projected foot (px)
  - ground plane: back-projected detection foot minus the 3D location (cx, cy) (m), binned by camera depth.
Reported separately for unoccluded persons (visible fraction >= 0.95) and for all matched persons. Post-hoc diagnostic on
development data; its output defines the NEG margin and ground cell size of protocol_v3 from measurements, not from scores.
Output: results/control_semantics/localization.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from analysis.control_gate import zones as Z
from analysis.edge_inference import evaluate as ev
from . import ground

OUT = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
ROLES = (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1))   # EXP-0010B holdouts are development data now
TAU = 0.3
DEPTH_BINS = [(0, 8), (8, 14), (14, 22), (22, 1e9)]
PCT = [50, 75, 90, 95]


def match_rows(clip, dets, tau, frames):
    fset = np.zeros(clip.n_frames, bool)
    fset[frames] = True
    keep = dets["score"] >= tau
    f, b = dets["frame"][keep], dets["box"][keep]
    order = np.argsort(f, kind="stable")
    f, b = f[order], b[order]
    st = np.searchsorted(f, np.arange(clip.n_frames), "left")
    en = np.searchsorted(f, np.arange(clip.n_frames), "right")
    sel = np.nonzero(fset[clip.frame] & (clip.vis_h >= Z.SIZE_FLOOR) & clip.depth_ok)[0]
    rows, dboxes = [], []
    for r in sel:
        fr = clip.frame[r]
        s, e = st[fr], en[fr]
        if e == s:
            continue
        v = clip.vis[r]
        if not (v[0] > 3 and v[1] > 3 and v[2] < 1917 and v[3] < 1077):
            continue
        iou = ev.box_iou(v[None], b[s:e])[0]
        j = int(np.argmax(iou))
        if iou[j] >= 0.5:
            rows.append(r)
            dboxes.append(b[s + j])
    return np.asarray(rows, int), np.asarray(dboxes)


def stats(x):
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), **{f"p{p}": round(float(np.percentile(x, p)), 3) for p in PCT}} if len(x) else {"n": 0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"tau": TAU, "note": "post-hoc diagnostic on development clips; matched = IoU>=0.5 vs the visible box", "clips": {}}
    for role, stride in ROLES:
        clip, _ = run_cag.load_clip(role)
        _, _, calp, cam_id = run_cag.paths(role)
        cam = gt3d.load_camera(calp, cam_id)
        dets, n = run_cag.load_dets(role, "YT", stride)
        frames = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
        rows, db = match_rows(clip, dets, TAU, frames)
        if len(rows) == 0:
            continue
        dfoot = np.stack([(db[:, 0] + db[:, 2]) / 2, db[:, 3]], 1)
        gfoot = clip.foot[rows]
        px_err = np.linalg.norm(dfoot - gfoot, axis=1)
        gxy_det, depth, ok = ground.pixel_to_ground(dfoot, cam)
        gt_xy = np.stack([clip.g["cx"][clip.prow[rows]], clip.g["cy"][clip.prow[rows]]], 1)
        m_err = np.linalg.norm(gxy_det - gt_xy, axis=1)
        m_err[~ok] = np.nan
        _, gdepth, _ = ground.pixel_to_ground(gfoot, cam)
        vf = clip.vis_frac[rows]
        res = {"matched_persons": int(len(rows)), "foot_error_px": {"unoccluded": stats(px_err[vf >= 0.95]), "all": stats(px_err)},
               "ground_error_m": {}, "metres_per_image_row_at_depth": {}}
        for lo, hi in DEPTH_BINS:
            sel = (gdepth >= lo) & (gdepth < hi)
            res["ground_error_m"][f"depth_{lo}-{hi if hi < 1e8 else 'inf'}m"] = {"unoccluded": stats(m_err[sel & (vf >= 0.95)]), "all": stats(m_err[sel])}
        res["ground_error_m"]["all_depths"] = {"unoccluded": stats(m_err[vf >= 0.95]), "all": stats(m_err)}
        for v in (300, 500, 700, 900):
            res["metres_per_image_row_at_depth"][f"v={v}"] = round(ground.metres_per_pixel_vertical(cam, v), 3)
        # how big are the 3x3 image-grid cells on the floor? (centre row pixels -> metres)
        xe, ye = Z.zone_edges()
        cells = {}
        for r_ in range(3):
            for c_ in range(3):
                cu, cv = (xe[c_] + xe[c_ + 1]) / 2, (ye[r_] + ye[r_ + 1]) / 2
                g, dd, ok1 = ground.pixel_to_ground(np.array([[xe[c_], cv], [xe[c_ + 1], cv], [cu, ye[r_]], [cu, ye[r_ + 1]]]), cam)
                cells[f"zone{r_ * 3 + c_}"] = None if not ok1.all() else {"width_m": round(float(np.linalg.norm(g[1] - g[0])), 1), "depth_m": round(float(np.linalg.norm(g[3] - g[2])), 1)}
        res["image_grid_cell_size_on_floor_m"] = cells
        out["clips"][role] = res
        print(role, "matched", len(rows), "px err p50/p90", res["foot_error_px"]["unoccluded"].get("p50"), res["foot_error_px"]["unoccluded"].get("p90"),
              "ground err (m) unoccluded all depths", res["ground_error_m"]["all_depths"]["unoccluded"], flush=True)
    (OUT / "localization.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
