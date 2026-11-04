"""Application-level quality metrics for a fixed camera: zone occupancy (primary), presence and count (secondary).

Definitions (preregistered in protocol_v2.json):
  * zones: GRID x GRID equal image cells over the 1920x1080 frame (an operator-drawn ROI stand-in).
  * a person's *position* is its ground-contact (foot) point in the image. For ground truth it is the projection
    of the 3D box bottom-centre (needs a validated calibration, see consistency.py); for a detection it is the
    bottom-centre of the predicted box. Neither depends on whether the box covers the visible or the whole person
    at the top, but a detection's bottom edge still depends on occlusion handling by the detector.
  * per (frame, zone): POS = a scored GT foot lies strictly inside the zone; NEG = no scored GT foot lies inside
    the zone grown by MARGIN px; otherwise DONTCARE (a person straddling the boundary: foot jitter is not an error).
    A frame-zone is *predicted occupied* if any detection foot with score >= tau lies inside the zone grown by MARGIN.
  * TPR = predicted-occupied among POS; FPR = predicted-occupied among NEG; balanced accuracy = (TPR + 1 - FPR)/2.
    Any constant predictor scores 0.5, so the metric cannot be won by a base-rate shortcut.
Scored GT persons: native visible height >= SIZE_FLOOR px (as in Stage A); smaller persons are ignore regions
(neither POS nor a reason for NEG, detections with feet there are discounted the same way as Stage A: a detection
shorter than SIZE_FLOOR is ignored).
"""
from __future__ import annotations

import numpy as np

W, H = 1920, 1080
GRID = 3
MARGIN = 32.0
SIZE_FLOOR = 32.0


def zone_edges(grid: int = GRID):
    return np.linspace(0, W, grid + 1), np.linspace(0, H, grid + 1)


def zone_masks(pts: np.ndarray, margin: float = 0.0, grid: int = GRID) -> np.ndarray:
    """(N,2) points -> (N, grid*grid) bool: point inside the zone grown by `margin` px (zone id = row*grid + col)."""
    xe, ye = zone_edges(grid)
    out = np.zeros((len(pts), grid * grid), dtype=bool)
    for r in range(grid):
        for c in range(grid):
            out[:, r * grid + c] = ((pts[:, 0] >= xe[c] - margin) & (pts[:, 0] < xe[c + 1] + margin) &
                                    (pts[:, 1] >= ye[r] - margin) & (pts[:, 1] < ye[r + 1] + margin))
    return out


def _by_frame(frames: np.ndarray, n: int):
    order = np.argsort(frames, kind="stable")
    fs = frames[order]
    start = np.searchsorted(fs, np.arange(n), side="left")
    end = np.searchsorted(fs, np.arange(n), side="right")
    return order, start, end


def gt_states(n_frames: int, foot: np.ndarray, frame: np.ndarray, height_px: np.ndarray, valid: np.ndarray,
              margin: float = MARGIN, grid: int = GRID):
    """-> (pos[n,Z], neg[n,Z]) bool. `valid` marks rows with a usable projected foot (in front of the camera)."""
    nz = grid * grid
    keep = valid & (height_px >= SIZE_FLOOR) & (foot[:, 0] >= 0) & (foot[:, 0] < W) & (foot[:, 1] >= 0) & (foot[:, 1] < H)
    strict = zone_masks(foot, 0.0, grid) & keep[:, None]
    grown = zone_masks(foot, margin, grid) & keep[:, None]
    pos = np.zeros((n_frames, nz), bool)
    anyg = np.zeros((n_frames, nz), bool)
    ok = frame < n_frames
    for z in range(nz):
        pos[frame[ok & strict[:, z]], z] = True
        anyg[frame[ok & grown[:, z]], z] = True
    return pos, ~anyg


def det_states(n_frames: int, dets: dict, tau: float, margin: float = MARGIN, grid: int = GRID) -> np.ndarray:
    """-> pred[n,Z] bool from detections dict {frame, box, score} (boxes native xyxy)."""
    nz = grid * grid
    sel = (dets["score"] >= tau)
    b = dets["box"][sel]
    f = dets["frame"][sel]
    tall = (b[:, 3] - b[:, 1]) >= SIZE_FLOOR
    b, f = b[tall], f[tall]
    foot = np.stack([(b[:, 0] + b[:, 2]) / 2, np.minimum(b[:, 3], H - 1e-3)], 1)
    grown = zone_masks(foot, margin, grid)
    pred = np.zeros((n_frames, nz), bool)
    ok = f < n_frames
    for z in range(nz):
        pred[f[ok & grown[:, z]], z] = True
    return pred


def occupancy_counts(pos, neg, pred, frames: np.ndarray | None = None) -> dict:
    sel = slice(None) if frames is None else frames
    p, n, d = pos[sel], neg[sel], pred[sel]
    tp, fn = int((p & d).sum()), int((p & ~d).sum())
    fp, tn = int((n & d).sum()), int((n & ~d).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def occupancy_metrics(c: dict) -> dict:
    tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
    fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
    prec = c["tp"] / max(c["tp"] + c["fp"], 1)
    return {"tpr": tpr, "fpr": fpr, "balanced_accuracy": (tpr + 1 - fpr) / 2, "precision": prec,
            "f1": 2 * prec * tpr / max(prec + tpr, 1e-12), "n_pos": c["tp"] + c["fn"], "n_neg": c["fp"] + c["tn"], **c}


def block_bootstrap_occupancy(pos, neg, pred, block: int = 300, n_boot: int = 1000, seed: int = 20261104) -> dict:
    n = len(pos)
    nb = (n + block - 1) // block
    bc = np.zeros((nb, 4), np.int64)
    for b in range(nb):
        s = slice(b * block, min((b + 1) * block, n))
        c = occupancy_counts(pos, neg, pred, np.arange(n)[s])
        bc[b] = (c["tp"], c["fn"], c["fp"], c["tn"])
    rng = np.random.default_rng(seed)
    tpr, fpr, ba = [], [], []
    for _ in range(n_boot):
        t = bc[rng.integers(0, nb, nb)].sum(0)
        a, bb = t[0] / max(t[0] + t[1], 1), t[2] / max(t[2] + t[3], 1)
        tpr.append(a); fpr.append(bb); ba.append((a + 1 - bb) / 2)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"tpr_ci": q(tpr), "fpr_ci": q(fpr), "balanced_accuracy_ci": q(ba), "block_frames": block, "n_boot": n_boot, "seed": seed}


def person_counts(n_frames, frame, height_px, valid, dets: dict | None = None, tau: float = 0.0):
    """Scored-person count per frame (GT) or detections >= tau with height >= SIZE_FLOOR (prediction)."""
    if dets is None:
        k = valid & (height_px >= SIZE_FLOOR)
        return np.bincount(frame[k], minlength=n_frames)[:n_frames]
    k = (dets["score"] >= tau) & ((dets["box"][:, 3] - dets["box"][:, 1]) >= SIZE_FLOOR)
    return np.bincount(dets["frame"][k], minlength=n_frames)[:n_frames]


def presence_and_count(gt_cnt: np.ndarray, pr_cnt: np.ndarray) -> dict:
    gp, pp = gt_cnt > 0, pr_cnt > 0
    return {"presence_recall": float((gp & pp).sum() / max(gp.sum(), 1)),
            "presence_false_alarm_rate": float((~gp & pp).sum() / max((~gp).sum(), 1)),
            "frames_without_scored_person": int((~gp).sum()),
            "count_mae": float(np.abs(gt_cnt - pr_cnt).mean()),
            "count_exact_frac": float((gt_cnt == pr_cnt).mean()),
            "count_within1_frac": float((np.abs(gt_cnt - pr_cnt) <= 1).mean()),
            "const_median_mae": float(np.abs(gt_cnt - np.median(gt_cnt)).mean())}
