"""Ground-plane zone-occupancy metric (protocol_v3 primary metric; numpy only).

Zones: square cells of CELL_M metres on the floor (world x,y), restricted to cells at least half inside the image and within
MAX_DEPTH_M of the camera (derived from calibration only). Positions: GT = the 3D box centre (cx, cy) of each agent; detection =
bottom-centre of the box back-projected to z=0. Tolerance MARGIN_M is the measured 95th-percentile ground localization error of
YOLOX-Tiny on development clips (results/control_semantics/localization.json), not a tuned value.

Per (frame, cell), for a monitoring POLICY that names the positive classes:
  POS        a scored positive-class agent lies strictly inside the cell
  NEG        no positive-class agent (scored OR ignored) lies within the cell grown by 2*MARGIN_M. A detection within MARGIN_M of the
             cell can only be explained by an agent within 2*MARGIN_M, so this is the geometric symmetric counterpart of the
             prediction tolerance. Non-positive agents (e.g. a humanoid under the human-only policy, a forklift) do NOT make a cell
             DONTCARE: a detection on them is a false alarm.
  DONTCARE   otherwise
Ignored agents (never scored, never a negative): visible height < SIZE_FLOOR_PX, visible box touching the frame edge (truncated),
projected foot outside the frame, or a bad projection. Detections that are shorter than SIZE_FLOOR_PX or touch the frame edge have
no reliable ground position and are excluded from the prediction (counted and reported, symmetric with ignored GT agents).
A frame-cell is predicted occupied if any retained detection (score >= tau) lies within the cell grown by MARGIN_M.
Metrics: TPR over POS, FPR over NEG, balanced accuracy; block bootstrap shared with the v2 metric.
"""
from __future__ import annotations

import numpy as np

from analysis.control_gate import zones as Z
from . import ground

CELL_M = 3.0
MARGIN_M = 0.75
MAX_DEPTH_M = 45.0
SIZE_FLOOR_PX = 32.0
EDGE_PX = 3.0
POLICIES = {"A_human_only": (0,), "B_person_shaped_agent": (0, 4, 5)}


def touches_edge(box: np.ndarray, edge: float = EDGE_PX) -> np.ndarray:
    return (box[:, 0] <= edge) | (box[:, 1] <= edge) | (box[:, 2] >= ground.W - edge) | (box[:, 3] >= ground.H - edge)


class CellLookup:
    """Vectorised (ix, iy) -> column lookup over the visible-cell set."""

    def __init__(self, cell_list: list):
        arr = np.asarray(cell_list, int).reshape(-1, 2)
        self.lo = arr.min(0) if len(arr) else np.zeros(2, int)
        shape = (arr.max(0) - self.lo + 1) if len(arr) else np.ones(2, int)
        self.grid = np.full(tuple(shape), -1, np.int64)
        for i, (ix, iy) in enumerate(arr):
            self.grid[ix - self.lo[0], iy - self.lo[1]] = i

    def get(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        gx, gy = ix - self.lo[0], iy - self.lo[1]
        ok = (gx >= 0) & (gy >= 0) & (gx < self.grid.shape[0]) & (gy < self.grid.shape[1])
        out = np.full(len(ix), -1, np.int64)
        out[ok] = self.grid[gx[ok], gy[ok]]
        return out


def _cells_for_points(xy: np.ndarray, reach: float, lookup: CellLookup, cell_m: float = CELL_M):
    """(point_idx, column) for every visible cell within `reach` metres (axis-aligned) of each point (reach <= cell_m)."""
    ix_lo = np.floor((xy[:, 0] - reach) / cell_m).astype(int)
    ix_hi = np.floor((xy[:, 0] + reach) / cell_m).astype(int)
    iy_lo = np.floor((xy[:, 1] - reach) / cell_m).astype(int)
    iy_hi = np.floor((xy[:, 1] + reach) / cell_m).astype(int)
    pis, cols = [], []
    for a in range(3):
        for b in range(3):
            ix, iy = ix_lo + a, iy_lo + b
            valid = (ix <= ix_hi) & (iy <= iy_hi)
            idx = np.nonzero(valid)[0]
            c = lookup.get(ix[idx], iy[idx])
            m = c >= 0
            pis.append(idx[m])
            cols.append(c[m])
    return np.concatenate(pis), np.concatenate(cols)


def gt_cell_states(n_frames: int, cam: dict, cell_list: list, g: dict, positive: tuple, margin: float = MARGIN_M, cell_m: float = CELL_M):
    """-> (pos, neg, info). g: control_gate.gt3d columns (all classes, visible boxes + 3D centres)."""
    lookup = CellLookup(cell_list)
    nc = len(cell_list)
    cls = g["cls"]
    vis = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1)
    xy = np.stack([g["cx"], g["cy"]], 1)
    uv, depth = ground.ground_to_pixel(xy, cam)
    inside = (depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < ground.W) & (uv[:, 1] >= 0) & (uv[:, 1] < ground.H)
    vh = vis[:, 3] - vis[:, 1]
    is_pos_cls = np.isin(cls, positive)
    scored = is_pos_cls & (vh >= SIZE_FLOOR_PX) & ~touches_edge(vis) & inside
    ignored = is_pos_cls & ~scored
    pos = np.zeros((n_frames, nc), bool)
    anyg = np.zeros((n_frames, nc), bool)
    fr = g["frame"]
    ok = fr < n_frames
    # POS: strictly inside the cell
    ix, iy = np.floor(xy[:, 0] / cell_m).astype(int), np.floor(xy[:, 1] / cell_m).astype(int)
    ks = np.nonzero(scored & ok)[0]
    c = lookup.get(ix[ks], iy[ks])
    m = c >= 0
    pos[fr[ks][m], c[m]] = True
    # NEG exclusion: any positive-class agent (scored or ignored) within 2*margin of the cell
    sel = np.nonzero(is_pos_cls & ok)[0]
    pi, col = _cells_for_points(xy[sel], 2 * margin, lookup, cell_m)
    anyg[fr[sel][pi], col] = True
    return pos, ~anyg, {"agents_scored": int(scored.sum()), "agents_ignored": int(ignored.sum())}


def det_cell_states(n_frames: int, cam: dict, cell_list: list, dets: dict, tau: float, margin: float = MARGIN_M, cell_m: float = CELL_M):
    lookup = CellLookup(cell_list)
    keep = dets["score"] >= tau
    b, f = dets["box"][keep], dets["frame"][keep]
    tall = (b[:, 3] - b[:, 1]) >= SIZE_FLOOR_PX
    edge = touches_edge(b)
    excluded_edge = int((tall & edge).sum())
    use = tall & ~edge & (f < n_frames)
    b, f = b[use], f[use]
    foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
    xy, depth, ok = ground.pixel_to_ground(foot, cam)
    pred = np.zeros((n_frames, len(cell_list)), bool)
    xy_ok, f_ok = xy[ok], f[ok]
    pi, col = _cells_for_points(xy_ok, margin, lookup, cell_m)
    pred[f_ok[pi], col] = True
    return pred, {"detections_used": int(use.sum()), "excluded_truncated_or_short": excluded_edge}


def metrics_for(pos, neg, pred, frames):
    """v2-style counts plus the cell-count-independent quantities used by protocol_v3.

    precision_cell = predicted-occupied cells that are POS / (POS + NEG predicted-occupied cells); DONTCARE cells are excluded.
    fa_per_predicted = 1 - precision_cell. frame_fa_rate = fraction of frames with at least one false-alarm (predicted & NEG) cell.
    FPR is kept for continuity with EXP-0010B but depends on how many empty cells a layout has.
    """
    m = Z.occupancy_counts(pos, neg, pred, frames)
    out = Z.occupancy_metrics(m)
    denom = m["tp"] + m["fp"]
    out["precision_cell"] = m["tp"] / denom if denom else float("nan")
    out["fa_per_predicted"] = m["fp"] / denom if denom else float("nan")
    fa_frames = (neg[frames] & pred[frames]).any(axis=1)
    out["frame_fa_rate"] = float(fa_frames.mean())
    out["frames_scored"] = int(len(frames))
    return out


def block_bootstrap(pos, neg, pred, frames, block: int = 300, n_boot: int = 1000, seed: int = 20261104) -> dict:
    """95% percentile block bootstrap of TPR, FPR, precision_cell and balanced accuracy over contiguous frame blocks."""
    fr = np.asarray(frames)
    nb = (len(fr) + block - 1) // block
    cnt = np.zeros((nb, 4), np.int64)
    for k in range(nb):
        sub = fr[k * block:(k + 1) * block]
        c = Z.occupancy_counts(pos, neg, pred, sub)
        cnt[k] = (c["tp"], c["fn"], c["fp"], c["tn"])
    rng = np.random.default_rng(seed)
    tpr, fpr, prec, ba = [], [], [], []
    for _ in range(n_boot):
        t = cnt[rng.integers(0, nb, nb)].sum(0)
        a, b = t[0] / max(t[0] + t[1], 1), t[2] / max(t[2] + t[3], 1)
        tpr.append(a); fpr.append(b); ba.append((a + 1 - b) / 2)
        prec.append(t[0] / (t[0] + t[2]) if t[0] + t[2] else np.nan)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"tpr_ci": q(tpr), "fpr_ci": q(fpr), "precision_cell_ci": q(prec), "balanced_accuracy_ci": q(ba),
            "block_frames": block, "n_boot": n_boot, "seed": seed}
