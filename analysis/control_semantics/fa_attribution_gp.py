"""POST-HOC attribution of policy-B false-alarm cells under the ground-plane metric (EXP-0010C; run after the fresh scoring).

Same method as fa_attribution.py (cause = the triggering detection's own box against GT boxes of every class, fixed priority),
but on the ground-plane cells at each clip's locked tau. Categories reuse fa_attribution.classify; 'person_scored' (a detection that
overlaps a scored Person yet lands in a NEG cell) is reported as person_position_mismatch (the box bottom is more than the margin from
the person's true floor position, e.g. an occluded or crouching person). Output: results/control_semantics/fa_attribution_gp.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from . import fa_attribution as FA
from . import gp_metric as GP
from . import ground, run_cs

RES = Path(__file__).resolve().parents[2] / "results" / "control_semantics"


def attribute(role, stride, tau, policy="B_person_shaped_agent"):
    clip, _ = run_cag.load_clip(role)
    _, _, calp, cam_id = run_cag.paths(role)
    cam = gt3d.load_camera(calp, cam_id)
    dets, n = run_cag.load_dets(role, "YT", stride)
    frames = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
    cells = ground.visible_cells(cam, GP.CELL_M, 0.5, GP.MAX_DEPTH_M)
    pos, neg, _ = GP.gt_cell_states(clip.n_frames, cam, cells, clip.g, GP.POLICIES[policy])
    keep = dets["score"] >= tau
    b, f, s = dets["box"][keep], dets["frame"][keep], dets["score"][keep]
    use = ((b[:, 3] - b[:, 1]) >= GP.SIZE_FLOOR_PX) & ~GP.touches_edge(b) & (f < clip.n_frames)
    b, f, s = b[use], f[use], s[use]
    xy, _, ok = ground.pixel_to_ground(np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1), cam)
    b, f, s, xy = b[ok], f[ok], s[ok], xy[ok]
    fset = np.zeros(clip.n_frames, bool)
    fset[frames] = True
    lookup = GP.CellLookup(cells)
    pi, col = GP._cells_for_points(xy, GP.MARGIN_M, lookup)
    sel = fset[f[pi]] & neg[f[pi], col]
    pi, col = pi[sel], col[sel]
    best = {}
    for k, c in zip(pi, col):
        key = (int(f[k]), int(c))
        if key not in best or s[k] > best[key][0]:
            best[key] = (float(s[k]), int(k))
    g = clip.g
    rows_by_f = {}
    for i, fr in enumerate(g["frame"]):
        rows_by_f.setdefault(int(fr), []).append(i)
    row_of = {int(r): k for k, r in enumerate(clip.prow)}
    counts = {}
    for (fr, c), (sc, k) in best.items():
        cat = FA.classify(b[k], sc, rows_by_f.get(fr, []), g, clip, row_of, None)
        cat = {"person_scored": "person_position_mismatch", "humanoid": "humanoid_position_mismatch"}.get(cat, cat)
        counts[cat] = counts.get(cat, 0) + 1
    n_fa = len(best)
    pred, _ = GP.det_cell_states(clip.n_frames, cam, cells, dets, tau)
    m = GP.metrics_for(pos, neg, pred, frames)
    return {"role": role, "tau": tau, "policy": policy, "false_alarm_cells": n_fa, "predicted_cells_scored": int(m["tp"] + m["fp"]),
            "fa_per_predicted": m["fa_per_predicted"], "cause_cells": counts,
            "cause_fraction_of_fa": {c: round(v / max(n_fa, 1), 4) for c, v in counts.items()}}


def main():
    lock = json.loads(run_cs.LOCK.read_text())
    out = {"note": "POST-HOC, run after the fresh scoring; policy B at the locked tau; fresh clips use the detections of the locked runs", "clips": []}
    run_cag.CLIPS.update(run_cs.fresh_clips())          # add hold_L05 / hold_L03 without touching the EXP-0010B role names
    tau = lock["tau"]["B_person_shaped_agent"]
    for role, stride in (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1), ("hold_L05", 1), ("hold_L03", 1)):
        out["clips"].append(attribute(role, stride, tau))
        c = out["clips"][-1]
        print(role, "FA/pred %.3f" % c["fa_per_predicted"], c["cause_fraction_of_fa"], flush=True)
    (RES / "fa_attribution_gp.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
