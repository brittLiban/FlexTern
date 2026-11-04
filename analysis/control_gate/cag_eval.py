"""Per-clip data loading and scoring for the control-adequacy gate (numpy only; runs on the Windows host too).

A ClipData holds the scene ground truth (visible boxes + 3D boxes), the calibration-derived foot points / expected
whole-person height, and the geometry-consistency verdict. `evaluate()` produces every metric the protocol names for one
(detections, tau, processed-frame subset).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.edge_inference import evaluate as ev
from . import consistency as C
from . import gt3d, zones

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)


@dataclass
class ClipData:
    name: str
    g: dict                      # parse_gt3d columns
    n_frames: int
    geometry_ok: bool
    consistency: dict
    prow: np.ndarray             # indices into g of Person rows
    vis: np.ndarray              # (P,4) visible boxes of persons
    foot: np.ndarray             # (P,2) projected foot point
    exp_h: np.ndarray            # (P,) expected whole-person pixel height
    depth_ok: np.ndarray         # (P,) bool
    aspect: float = 0.4
    cache: dict = field(default_factory=dict)

    @property
    def vis_h(self):
        return self.vis[:, 3] - self.vis[:, 1]

    @property
    def vis_frac(self):
        return np.where(self.exp_h > 1, self.vis_h / np.maximum(self.exp_h, 1e-6), np.nan)

    @property
    def frame(self):
        return self.g["frame"][self.prow]


def build_clip(name: str, g: dict, n_frames: int, cam: dict) -> ClipData:
    cons = C.scene_consistency(g, cam)
    ok = C.passes(cons)
    prow, vis, foot, eh, dz = C.person_geometry(g, cam, "box_bottom")
    vh = vis[:, 3] - vis[:, 1]
    vw = vis[:, 2] - vis[:, 0]
    good = (eh > 20) & (vh / np.maximum(eh, 1e-6) >= 0.95) & (vh / np.maximum(eh, 1e-6) <= 1.1) & C.interior(vis, dz)
    aspect = float(np.median(vw[good] / vh[good])) if good.sum() >= 50 else 0.4
    return ClipData(name, g, n_frames, ok, cons, prow, vis, foot, eh, dz > 0.1, aspect)


def _gt_dict(clip: ClipData, whole: bool = False) -> dict:
    g = {k: clip.g[k].copy() for k in ("frame", "x1", "y1", "x2", "y2", "cls")}
    if whole:
        h = np.maximum(clip.exp_h, 1.0)
        w = clip.aspect * h
        r = clip.prow
        g["x1"][r] = clip.foot[:, 0] - w / 2
        g["x2"][r] = clip.foot[:, 0] + w / 2
        g["y1"][r] = clip.foot[:, 1] - h
        g["y2"][r] = clip.foot[:, 1]
    return g


def _iou_matrix_boxes(a, b):
    return ev.box_iou(a, b)


def ap_block(clip: ClipData, dets: dict, tau: float, frames: np.ndarray) -> dict:
    out = {}
    for thr in (0.1, 0.3, 0.5):
        cs = ev.score_clip(clip.n_frames, dets, _gt_dict(clip), iou_thr=thr, frames=frames)
        n_gt = int((~cs.gt_ignore).sum())
        out[f"ap_iou{thr}"] = ev.ap_from_arrays(cs.det_score, cs.det_flag, None, n_gt)
        if thr == 0.5:
            op = ev.operating_point(cs, tau)
            out["precision_iou0.5"], out["recall_iou0.5"], out["f1_iou0.5"] = op["precision"], op["recall"], op["f1"]
    if clip.geometry_ok:   # INFERRED whole-person boxes from calibration + median unoccluded aspect
        for thr in (0.5,):
            cs = ev.score_clip(clip.n_frames, dets, _gt_dict(clip, whole=True), iou_thr=thr, frames=frames)
            out["ap_iou0.5_vs_whole_person_box_INFERRED"] = ev.ap_from_arrays(cs.det_score, cs.det_flag, None, int((~cs.gt_ignore).sum()))
    return out


def zone_block(clip: ClipData, dets: dict, tau: float, frames: np.ndarray, boot: bool = True) -> dict | None:
    if not clip.geometry_ok:
        return None
    n = clip.n_frames
    pos, neg = zones.gt_states(n, clip.foot, clip.frame, clip.vis_h, clip.depth_ok)
    pred = zones.det_states(n, dets, tau)
    c = zones.occupancy_counts(pos, neg, pred, frames)
    out = zones.occupancy_metrics(c)
    out["per_zone_pos_rate"] = pos[frames].mean(0).round(4).tolist()
    out["per_zone_neg_rate"] = neg[frames].mean(0).round(4).tolist()
    if boot:
        sub = np.asarray(frames)
        out["bootstrap"] = zones.block_bootstrap_occupancy(pos[sub], neg[sub], pred[sub])
    return out


def zone_counts_only(clip: ClipData, dets: dict, tau_grid, frames: np.ndarray) -> dict:
    """Fast: confusion counts for each tau (used for tau selection and pooled dev selection)."""
    if not clip.geometry_ok:
        return {}
    n = clip.n_frames
    key = "gt_states"
    if key not in clip.cache:
        clip.cache[key] = zones.gt_states(n, clip.foot, clip.frame, clip.vis_h, clip.depth_ok)
    pos, neg = clip.cache[key]
    return {float(t): zones.occupancy_counts(pos, neg, zones.det_states(n, dets, float(t)), frames) for t in tau_grid}


def presence_count_block(clip: ClipData, dets: dict, tau: float, frames: np.ndarray) -> dict:
    n = clip.n_frames
    valid = clip.depth_ok if clip.geometry_ok else np.ones(len(clip.prow), bool)
    gt_cnt = zones.person_counts(n, clip.frame, clip.vis_h, valid)[frames]
    pr_cnt = zones.person_counts(n, None, None, None, dets, tau)[frames]
    return zones.presence_and_count(gt_cnt, pr_cnt)


def visibility_block(clip: ClipData, dets: dict, tau: float, frames: np.ndarray, iou_thr=(0.3, 0.5)) -> dict | None:
    """Per-GT-person detection hit rate by visible fraction, and box-extent errors of the best-overlapping detection."""
    if not clip.geometry_ok:
        return None
    fset = np.zeros(clip.n_frames, bool)
    fset[frames] = True
    sel = fset[clip.frame] & (clip.vis_h >= zones.SIZE_FLOOR) & clip.depth_ok & (clip.exp_h > 20)
    rows = np.nonzero(sel)[0]
    keep = dets["score"] >= tau
    dframe, dbox = dets["frame"][keep], dets["box"][keep]
    order = np.argsort(dframe, kind="stable")
    dframe, dbox = dframe[order], dbox[order]
    starts = np.searchsorted(dframe, np.arange(clip.n_frames), side="left")
    ends = np.searchsorted(dframe, np.arange(clip.n_frames), side="right")
    best_iou = np.zeros(len(rows))
    top_err = np.full(len(rows), np.nan)
    bot_err = np.full(len(rows), np.nan)
    hrat = np.full(len(rows), np.nan)
    f_rows = clip.frame[rows]
    for i, (r, f) in enumerate(zip(rows, f_rows)):
        s, e = starts[f], ends[f]
        if e == s:
            continue
        ious = ev.box_iou(clip.vis[r:r + 1], dbox[s:e])[0]
        j = int(np.argmax(ious))
        best_iou[i] = ious[j]
        if ious[j] >= 0.1:
            d = dbox[s + j]
            top_err[i] = (d[1] - clip.vis[r, 1]) / clip.exp_h[r]       # + = detection starts lower than visible top
            bot_err[i] = (d[3] - clip.vis[r, 3]) / clip.exp_h[r]
            hrat[i] = (d[3] - d[1]) / clip.exp_h[r]
    vf = clip.vis_frac[rows]
    bins = [("<0.5", vf < 0.5), ("0.5-0.8", (vf >= 0.5) & (vf < 0.8)), ("0.8-0.95", (vf >= 0.8) & (vf < 0.95)), (">=0.95", vf >= 0.95)]
    out = {"n_scored_gt": int(len(rows)), "bins": {}}
    for name, m in bins:
        k = int(m.sum())
        b = {"n": k}
        if k:
            b["hit_iou>=0.5"] = float((best_iou[m] >= 0.5).mean())
            b["hit_iou>=0.3"] = float((best_iou[m] >= 0.3).mean())
            b["hit_iou>=0.1"] = float((best_iou[m] >= 0.1).mean())
            ok = m & ~np.isnan(hrat)
            if ok.sum():
                b["median_det_height_over_expected_whole_height"] = float(np.median(hrat[ok]))
                b["median_top_err_over_expected_h"] = float(np.median(top_err[ok]))
                b["median_bottom_err_over_expected_h"] = float(np.median(bot_err[ok]))
        out["bins"][name] = b
    return out


def evaluate(clip: ClipData, dets: dict, tau: float, frames: np.ndarray, boot: bool = True) -> dict:
    return {"clip": clip.name, "tau": tau, "n_frames_scored": int(len(frames)), "geometry_ok": clip.geometry_ok,
            "zone": zone_block(clip, dets, tau, frames, boot), "presence_count": presence_count_block(clip, dets, tau, frames),
            "ap": ap_block(clip, dets, tau, frames), "visibility": visibility_block(clip, dets, tau, frames)}
