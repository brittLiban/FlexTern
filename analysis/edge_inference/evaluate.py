"""Single-class detection scoring for Stage A (numpy only).

- COCO-style greedy matching per frame at a fixed IoU, with *ignore* GT regions (small
  persons, humanoid robots) and ignored small unmatched detections.
- AP50 (101-point interpolation), precision/recall/F1 at an operating threshold, per-size
  recall, frame-level occupancy metrics.
- Block bootstrap over contiguous segments (frames are temporally correlated, so frames
  are NOT resampled independently).

Detections for a clip are passed as flat arrays plus a frame index. Dropped/unprocessed
frames simply have no detections, so "quality including drops" falls out naturally.
"""
from __future__ import annotations

import numpy as np

SMALL_H = 32.0


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def match_frame(det_boxes, det_scores, gt_boxes, gt_ignore, iou_thr=0.5, small_h=SMALL_H):
    """Greedy COCO matching for one frame.

    Returns (flags, gt_matched_score):
      flags[d] = 1 TP, 0 FP, -1 ignored (matched an ignore GT, or unmatched and shorter than small_h)
      gt_matched_score[g] = score of the detection that matched GT g (non-ignore GTs), else -1
    Detections must already be sorted by descending score.
    """
    nd, ng = len(det_boxes), len(gt_boxes)
    flags = np.zeros(nd, dtype=np.int8)
    gms = np.full(ng, -1.0, dtype=np.float64)
    if nd == 0:
        return flags, gms
    det_h = det_boxes[:, 3] - det_boxes[:, 1]
    if ng == 0:
        flags[det_h < small_h] = -1
        return flags, gms
    iou = box_iou(det_boxes, gt_boxes)
    gt_ignore = np.asarray(gt_ignore, dtype=bool)
    order = np.argsort(gt_ignore, kind="stable")  # non-ignore GTs first (COCO preference)
    taken = np.zeros(ng, dtype=bool)
    thr0 = min(iou_thr, 1 - 1e-10)
    cand_set = set(np.nonzero(iou.max(axis=1) >= thr0)[0].tolist())
    for d in range(nd):
        if d not in cand_set:
            if det_h[d] < small_h:
                flags[d] = -1
            continue
        best, thr = -1, thr0
        for g in order:
            if taken[g]:
                continue
            if best > -1 and not gt_ignore[best] and gt_ignore[g]:
                break
            if iou[d, g] < thr:
                continue
            thr, best = iou[d, g], g
        if best == -1:
            flags[d] = -1 if det_h[d] < small_h else 0
        elif gt_ignore[best]:
            taken[best] = True
            flags[d] = -1
        else:
            taken[best] = True
            flags[d] = 1
            gms[best] = det_scores[d]
    return flags, gms


class ClipScore:
    """Per-detection / per-GT match results for one clip; the basis for all metrics."""

    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self.det_frame: np.ndarray
        self.det_score: np.ndarray
        self.det_flag: np.ndarray
        self.gt_frame: np.ndarray
        self.gt_height: np.ndarray
        self.gt_ignore: np.ndarray
        self.gt_matched_score: np.ndarray


def score_clip(n_frames: int, dets: dict, gt: dict, person_cls: int = 0, iou_thr: float = 0.5,
               small_h: float = SMALL_H, ignore_classes: tuple[int, ...] = (4, 5), frame_shift: int = 0,
               frames: np.ndarray | None = None) -> ClipScore:
    """dets: {'frame','score','box'[N,4]} flat; gt: gt.load_camera_gt dict.

    frame_shift: score detections of frame f against GT of frame f+frame_shift (alignment sweep only).
    frames: score only these frame indices (e.g. a stride subset); default all frames.
    """
    dmap = _group(dets["frame"], n_frames)
    gcls = gt["cls"]
    sel = (gcls == person_cls) | np.isin(gcls, ignore_classes)
    gf = gt["frame"][sel]
    gb = np.stack([gt["x1"][sel], gt["y1"][sel], gt["x2"][sel], gt["y2"][sel]], axis=1).astype(np.float64)
    gc = gcls[sel]
    gh = gb[:, 3] - gb[:, 1]
    gig = (gc != person_cls) | (gh < small_h)
    gmap = _group(gf, n_frames + abs(frame_shift) + 1)
    out_df, out_ds, out_dfl, out_gf, out_gh, out_gig, out_gms = [], [], [], [], [], [], []
    for f in (range(n_frames) if frames is None else [int(x) for x in frames]):
        di = dmap.get(f)
        gi = gmap.get(f + frame_shift, np.zeros(0, dtype=np.int64))
        db = dets["box"][di] if di is not None else np.zeros((0, 4))
        ds = dets["score"][di] if di is not None else np.zeros(0)
        if len(ds):
            o = np.argsort(-ds, kind="stable")
            db, ds = db[o], ds[o]
        flags, gms = match_frame(db, ds, gb[gi], gig[gi], iou_thr, small_h)
        out_df.append(np.full(len(ds), f, np.int32))
        out_ds.append(ds)
        out_dfl.append(flags)
        out_gf.append(np.full(len(gi), f, np.int32))
        out_gh.append(gh[gi])
        out_gig.append(gig[gi])
        out_gms.append(gms)
    cs = ClipScore(n_frames)
    cs.det_frame = np.concatenate(out_df) if out_df else np.zeros(0, np.int32)
    cs.det_score = np.concatenate(out_ds) if out_ds else np.zeros(0)
    cs.det_flag = np.concatenate(out_dfl) if out_dfl else np.zeros(0, np.int8)
    cs.gt_frame = np.concatenate(out_gf) if out_gf else np.zeros(0, np.int32)
    cs.gt_height = np.concatenate(out_gh) if out_gh else np.zeros(0)
    cs.gt_ignore = np.concatenate(out_gig) if out_gig else np.zeros(0, bool)
    cs.gt_matched_score = np.concatenate(out_gms) if out_gms else np.zeros(0)
    return cs


def _group(frames: np.ndarray, n: int) -> dict:
    if len(frames) == 0:
        return {}
    order = np.argsort(frames, kind="stable")
    fs = frames[order]
    cuts = np.nonzero(np.diff(fs))[0] + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(fs)]])
    return {int(fs[s]): order[s:e] for s, e in zip(starts, ends)}


# ---------------------------------------------------------------- metrics

def ap_from_arrays(score: np.ndarray, flag: np.ndarray, weight: np.ndarray | None, n_gt: float) -> float:
    """101-point interpolated AP. `weight` allows bootstrap block multiplicities."""
    keep = flag >= 0
    s, f = score[keep], flag[keep]
    w = np.ones(len(s)) if weight is None else weight[keep]
    if n_gt <= 0 or len(s) == 0:
        return float("nan") if n_gt <= 0 else 0.0
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(w[o] * (f[o] == 1))
    fp = np.cumsum(w[o] * (f[o] == 0))
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    thr = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, thr, side="left")
    q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(q.mean())


def operating_point(cs: ClipScore, tau: float, strata: dict | None = None) -> dict:
    valid_gt = ~cs.gt_ignore
    n_gt = int(valid_gt.sum())
    d = cs.det_score >= tau
    tp = int(((cs.det_flag == 1) & d).sum())
    fp = int(((cs.det_flag == 0) & d).sum())
    gm = valid_gt & (cs.gt_matched_score >= tau)
    rec = gm.sum() / n_gt if n_gt else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else 0.0
    res = {"tau": tau, "n_gt": n_gt, "tp": tp, "fp": fp, "fn": n_gt - int(gm.sum()),
           "precision": prec, "recall": float(rec), "f1": float(f1)}
    for name, (lo, hi) in (strata or {}).items():
        m = (cs.gt_height >= lo) & (cs.gt_height < hi)
        # strata are reported on all-size GT (ignore flag only governs the primary metric)
        n = int(m.sum())
        res[f"recall_{name}"] = float((cs.gt_matched_score[m] >= tau).sum() / n) if n else float("nan")
        res[f"n_gt_{name}"] = n
    # frame-level occupancy: any valid person present vs any TP at tau
    has_gt = np.zeros(cs.n_frames, bool)
    has_gt[cs.gt_frame[valid_gt]] = True
    has_tp = np.zeros(cs.n_frames, bool)
    has_tp[cs.det_frame[(cs.det_flag == 1) & d]] = True
    has_fp = np.zeros(cs.n_frames, bool)
    has_fp[cs.det_frame[(cs.det_flag == 0) & d]] = True
    res["frames_with_person"] = int(has_gt.sum())
    res["frame_recall"] = float((has_gt & has_tp).sum() / has_gt.sum()) if has_gt.any() else float("nan")
    res["frame_false_alarm_rate"] = float((~has_gt & has_fp).sum() / (~has_gt).sum()) if (~has_gt).any() else float("nan")
    return res


def clip_metrics(cs: ClipScore, tau: float, strata: dict | None = None) -> dict:
    n_gt = int((~cs.gt_ignore).sum())
    out = {"ap50": ap_from_arrays(cs.det_score, cs.det_flag, None, n_gt)}
    out.update(operating_point(cs, tau, strata))
    return out


def best_f1_threshold(cs: ClipScore, grid=None) -> tuple[float, list]:
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2) if grid is None else grid
    rows = [operating_point(cs, float(t)) for t in grid]
    best = max(rows, key=lambda r: (r["f1"], -r["tau"]))
    return best["tau"], rows


def block_bootstrap(cs: ClipScore, tau: float, block: int = 300, n_boot: int = 1000, seed: int = 20261104) -> dict:
    """Percentile 95% CIs for AP50 / recall / precision / F1, resampling contiguous frame blocks."""
    nb = int(np.ceil(cs.n_frames / block))
    dblk = np.minimum(cs.det_frame // block, nb - 1)
    gblk = np.minimum(cs.gt_frame // block, nb - 1)
    valid = ~cs.gt_ignore
    gt_per = np.bincount(gblk[valid], minlength=nb).astype(float)
    tp_det = (cs.det_flag == 1) & (cs.det_score >= tau)
    fp_det = (cs.det_flag == 0) & (cs.det_score >= tau)
    tp_per = np.bincount(dblk[tp_det], minlength=nb).astype(float)
    fp_per = np.bincount(dblk[fp_det], minlength=nb).astype(float)
    gm_per = np.bincount(gblk[valid & (cs.gt_matched_score >= tau)], minlength=nb).astype(float)
    rng = np.random.default_rng(seed)
    aps, recs, precs, f1s = [], [], [], []
    for _ in range(n_boot):
        w_blk = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        n_gt = float((w_blk * gt_per).sum())
        aps.append(ap_from_arrays(cs.det_score, cs.det_flag, w_blk[dblk], n_gt))
        tp, fp, gm = (w_blk * tp_per).sum(), (w_blk * fp_per).sum(), (w_blk * gm_per).sum()
        r = gm / n_gt if n_gt else np.nan
        p = tp / (tp + fp) if tp + fp else np.nan
        recs.append(r)
        precs.append(p)
        f1s.append(2 * p * r / (p + r) if (p + r) and not np.isnan(p + r) else 0.0)

    def ci(v):
        v = np.asarray(v, float)
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {"block_frames": block, "n_blocks": nb, "n_boot": n_boot, "seed": seed,
            "ap50_ci95": ci(aps), "recall_ci95": ci(recs), "precision_ci95": ci(precs), "f1_ci95": ci(f1s)}
