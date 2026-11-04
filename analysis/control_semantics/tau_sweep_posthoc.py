"""POST-HOC threshold sweep for policy B on all six clips (EXP-0010C). Run after the fresh scoring; does not change any verdict.

Asks: is there ANY score threshold at which YOLOX-Tiny meets the gating pair (TPR >= 0.90, false alarms per predicted cell <= 0.10)?
Dev clips use their stride-3 detections (L10, L11) or full detections (L08, L02); the fresh clips use the locked runs' detections.
Output: results/control_semantics/tau_sweep_posthoc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from . import gp_metric as GP
from . import ground, run_cs

RES = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
TAUS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CLIPS = (("dev_L10", 3), ("dev_L11", 3), ("hold_L08", 1), ("hold_L02", 1), ("hold_L05", 1), ("hold_L03", 1))


def main():
    run_cag.CLIPS.update(run_cs.fresh_clips())
    out = {"policy": "B_person_shaped_agent", "taus": TAUS, "note": "POST-HOC after the fresh scoring", "clips": {}}
    for role, stride in CLIPS:
        clip, _ = run_cag.load_clip(role)
        _, _, calp, cam_id = run_cag.paths(role)
        cam = gt3d.load_camera(calp, cam_id)
        dets, n = run_cag.load_dets(role, "YT", stride)
        frames = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
        cells = ground.visible_cells(cam, GP.CELL_M, 0.5, GP.MAX_DEPTH_M)
        pos, neg, _ = GP.gt_cell_states(clip.n_frames, cam, cells, clip.g, GP.POLICIES["B_person_shaped_agent"])
        rows = []
        for t in TAUS:
            pred, _ = GP.det_cell_states(clip.n_frames, cam, cells, dets, t)
            m = GP.metrics_for(pos, neg, pred, frames)
            rows.append({"tau": t, "tpr": round(m["tpr"], 3), "fa_per_predicted": round(m["fa_per_predicted"], 3), "fpr": round(m["fpr"], 3)})
        out["clips"][role] = rows
        print(role, [(r["tau"], r["tpr"], r["fa_per_predicted"]) for r in rows], flush=True)
    any_tau = [t for t in TAUS if all(next(r for r in out["clips"][c] if r["tau"] == t)["tpr"] >= 0.90 and
                                      next(r for r in out["clips"][c] if r["tau"] == t)["fa_per_predicted"] <= 0.10 for c, _ in CLIPS)]
    out["taus_meeting_the_pair_on_all_six_clips"] = any_tau
    per_clip_any = {c: [r["tau"] for r in out["clips"][c] if r["tpr"] >= 0.90 and r["fa_per_predicted"] <= 0.10] for c, _ in CLIPS}
    out["taus_meeting_the_pair_per_clip"] = per_clip_any
    print("meeting the pair on all six:", any_tau, "per clip:", per_clip_any)
    (RES / "tau_sweep_posthoc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
