"""Agent-level recall of YOLOX-Tiny on development clips by visibility and size (EXP-0010C, post-hoc diagnostic).

For every scored Person (visible height >= 32 px, not touching the frame edge, valid ground position) is there a detection of
score >= TAU within MARGIN_M (ground distance) of its 3D position? Reported by visible fraction (visible height / expected
whole-person height, from calibration; scenes passing the geometry rule) and by visible height. Separates occlusion-limited
and size-limited misses from clean misses. Output: results/control_semantics/miss_attribution.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from . import gp_metric as GP
from . import ground

OUT = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
ROLES = (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1))
TAU = 0.02


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"tau": TAU, "margin_m": GP.MARGIN_M, "clips": {}}
    for role, stride in ROLES:
        clip, _ = run_cag.load_clip(role)
        _, _, calp, cam_id = run_cag.paths(role)
        cam = gt3d.load_camera(calp, cam_id)
        dets, n = run_cag.load_dets(role, "YT", stride)
        frames = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
        fset = np.zeros(clip.n_frames, bool)
        fset[frames] = True
        keep = (dets["score"] >= TAU) & ((dets["box"][:, 3] - dets["box"][:, 1]) >= GP.SIZE_FLOOR_PX) & ~GP.touches_edge(dets["box"])
        f, b = dets["frame"][keep], dets["box"][keep]
        xy, _, ok = ground.pixel_to_ground(np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1), cam)
        f, xy = f[ok], xy[ok]
        order = np.argsort(f, kind="stable")
        f, xy = f[order], xy[order]
        st, en = np.searchsorted(f, np.arange(clip.n_frames), "left"), np.searchsorted(f, np.arange(clip.n_frames), "right")
        rows = np.nonzero(fset[clip.frame] & (clip.vis_h >= GP.SIZE_FLOOR_PX) & clip.depth_ok & ~GP.touches_edge(clip.vis))[0]
        gxy = np.stack([clip.g["cx"][clip.prow[rows]], clip.g["cy"][clip.prow[rows]]], 1)
        hit = np.zeros(len(rows), bool)
        for i, (r, p) in enumerate(zip(rows, gxy)):
            s, e = st[clip.frame[r]], en[clip.frame[r]]
            if e > s and np.min(np.linalg.norm(xy[s:e] - p, axis=1)) <= GP.MARGIN_M:
                hit[i] = True
        vf, vh = clip.vis_frac[rows], clip.vis_h[rows]
        res = {"scored_persons": int(len(rows)), "recall_all": float(hit.mean()), "by_visible_fraction": {}, "by_visible_height_px": {}}
        for name, m in (("<0.5", vf < 0.5), ("0.5-0.8", (vf >= 0.5) & (vf < 0.8)), ("0.8-0.95", (vf >= 0.8) & (vf < 0.95)), (">=0.95", vf >= 0.95)):
            res["by_visible_fraction"][name] = {"n": int(m.sum()), "recall": float(hit[m].mean()) if m.any() else None,
                                                "share_of_misses": float((~hit & m).sum() / max((~hit).sum(), 1))}
        for name, m in (("32-64", (vh >= 32) & (vh < 64)), ("64-128", (vh >= 64) & (vh < 128)), (">=128", vh >= 128)):
            res["by_visible_height_px"][name] = {"n": int(m.sum()), "recall": float(hit[m].mean()) if m.any() else None,
                                                 "share_of_misses": float((~hit & m).sum() / max((~hit).sum(), 1))}
        out["clips"][role] = res
        print(role, round(res["recall_all"], 3), {k: (v["n"], round(v["recall"], 2) if v["recall"] is not None else None, round(v["share_of_misses"], 2)) for k, v in res["by_visible_fraction"].items()},
              {k: (v["n"], round(v["recall"], 2) if v["recall"] is not None else None) for k, v in res["by_visible_height_px"].items()}, flush=True)
    (OUT / "miss_attribution.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
