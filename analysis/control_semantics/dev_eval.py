"""Development evaluation of YOLOX-Tiny under the ground-plane metric for both monitoring policies (EXP-0010C).

Development clips: L10, L11 (no humanoids) and L08, L02 (humanoids in every frame; inspected during EXP-0010B so they are development
data now). Per policy the operating threshold is the tau maximising the MACRO mean (over development clips) balanced accuracy on
a grid that extends below 0.05. Output: results/control_semantics/dev_policy_scores.json (and the tau choices used by protocol_v3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from analysis.control_gate import zones as Z
from . import gp_metric as GP
from . import ground

OUT = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
ROLES = (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1))
TAUS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def clip_gp(role, stride, config="YT"):
    clip, _ = run_cag.load_clip(role)
    _, _, calp, cam_id = run_cag.paths(role)
    cam = gt3d.load_camera(calp, cam_id)
    dets, n = run_cag.load_dets(role, config, stride)
    frames = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
    cells = ground.visible_cells(cam, GP.CELL_M, 0.5, GP.MAX_DEPTH_M)
    return clip, cam, dets, frames, cells


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cell_m": GP.CELL_M, "margin_m": GP.MARGIN_M, "taus": TAUS, "policies": {}, "clips": {}}
    cache = {}
    for role, stride in ROLES:
        clip, cam, dets, frames, cells = clip_gp(role, stride)
        gts = {pol: GP.gt_cell_states(clip.n_frames, cam, cells, clip.g, cls) for pol, cls in GP.POLICIES.items()}
        cache[role] = (clip, cam, dets, frames, cells, gts)
        res["clips"][role] = {"n_cells": len(cells), "frames": int(len(frames)),
                              "gt_info": {pol: gts[pol][2] for pol in gts},
                              "pos_rate_A": float(gts["A_human_only"][0][frames].mean()), "pos_rate_B": float(gts["B_person_shaped_agent"][0][frames].mean())}
        print(role, "cells", len(cells), res["clips"][role], flush=True)
    for pol in GP.POLICIES:
        grid = {}
        for tau in TAUS:
            per = {}
            for role, _ in ROLES:
                clip, cam, dets, frames, cells, gts = cache[role]
                pred, info = GP.det_cell_states(clip.n_frames, cam, cells, dets, tau)
                pos, neg, _ = gts[pol]
                m = GP.metrics_for(pos, neg, pred, frames)
                per[role] = {"tpr": m["tpr"], "fpr": m["fpr"], "ba": m["balanced_accuracy"], "precision_cell": m["precision_cell"], "frame_fa_rate": m["frame_fa_rate"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                             "excluded_truncated_or_short_dets": info["excluded_truncated_or_short"]}
            grid[tau] = {"per_clip": per, "macro_ba": float(np.mean([v["ba"] for v in per.values()]))}
        best = max(TAUS, key=lambda t: (grid[t]["macro_ba"], -t))
        res["policies"][pol] = {"best_tau": best, "macro_ba_at_best": grid[best]["macro_ba"], "grid": {str(k): v for k, v in grid.items()}}
        print(pol, "best tau", best, "macro BA %.3f" % grid[best]["macro_ba"],
              {r: (round(v["tpr"], 3), round(v["fpr"], 3), round(v["ba"], 3)) for r, v in grid[best]["per_clip"].items()}, flush=True)
        # bootstrap CIs at the chosen tau
        res["policies"][pol]["bootstrap_at_best"] = {}
        for role, _ in ROLES:
            clip, cam, dets, frames, cells, gts = cache[role]
            pred, _ = GP.det_cell_states(clip.n_frames, cam, cells, dets, best)
            pos, neg, _ = gts[pol]
            res["policies"][pol]["bootstrap_at_best"][role] = GP.block_bootstrap(pos, neg, pred, frames)
    (OUT / "dev_policy_scores.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
