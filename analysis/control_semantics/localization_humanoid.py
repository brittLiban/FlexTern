"""Humanoid ground localization of YOLOX-Tiny (POST-HOC, EXP-0010C, after the advisor's completion review).

The 0.75 m margin of protocol_v3 was measured on Person rows only (localization.py), but policy B scores humanoids with it. This repeats the
measurement for FourierGR1T2 / AgilityDigit on the development clips that contain humanoids (L08, L02): scored humanoids (visible height >= 32 px,
box wholly inside the frame) matched to a detection of score >= 0.3 with IoU >= 0.5, detection foot (box bottom-centre) back-projected to the floor
vs the 3D location. Output: results/control_semantics/localization_humanoid.json (the locked margin is NOT changed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d, run_cag
from analysis.edge_inference import evaluate as ev
from . import ground

OUT = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
TAU = 0.3
PCT = [50, 75, 90, 95]


def main():
    res = {"tau": TAU, "locked_margin_m": 0.75, "clips": {}}
    for role in ("hold_L08", "hold_L02"):
        clip, _ = run_cag.load_clip(role)
        g = clip.g
        _, _, calp, cam_id = run_cag.paths(role)
        cam = gt3d.load_camera(calp, cam_id)
        dets, n = run_cag.load_dets(role, "YT", 1)
        keep = dets["score"] >= TAU
        f, b = dets["frame"][keep], dets["box"][keep]
        order = np.argsort(f, kind="stable")
        f, b = f[order], b[order]
        st, en = np.searchsorted(f, np.arange(clip.n_frames), "left"), np.searchsorted(f, np.arange(clip.n_frames), "right")
        for name, ids in (("FourierGR1T2", (4,)), ("AgilityDigit", (5,)), ("person_reference", (0,))):
            rows = np.nonzero(np.isin(g["cls"], ids) & (g["frame"] < n))[0]
            vis = np.stack([g[k][rows] for k in ("x1", "y1", "x2", "y2")], 1)
            inside = (vis[:, 0] > 3) & (vis[:, 1] > 3) & (vis[:, 2] < 1917) & (vis[:, 3] < 1077) & ((vis[:, 3] - vis[:, 1]) >= 32)
            errs, dbs = [], []
            for r, v in zip(rows[inside], vis[inside]):
                s, e = st[g["frame"][r]], en[g["frame"][r]]
                if e == s:
                    continue
                iou = ev.box_iou(v[None], b[s:e])[0]
                j = int(np.argmax(iou))
                if iou[j] >= 0.5:
                    dbs.append(b[s + j])
                    errs.append(r)
            if not errs:
                continue
            db = np.asarray(dbs)
            xy, depth, ok = ground.pixel_to_ground(np.stack([(db[:, 0] + db[:, 2]) / 2, db[:, 3]], 1), cam)
            gt_xy = np.stack([g["cx"][errs], g["cy"][errs]], 1)
            e_m = np.linalg.norm(xy - gt_xy, axis=1)[ok]
            res["clips"].setdefault(role, {})[name] = {"matched": int(len(e_m)), **{f"p{p}": round(float(np.percentile(e_m, p)), 3) for p in PCT},
                                                       "frac_beyond_locked_margin": round(float((e_m > 0.75).mean()), 4)}
            print(role, name, res["clips"][role][name], flush=True)
    (OUT / "localization_humanoid.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
