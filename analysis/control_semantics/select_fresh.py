"""Choose the fresh holdout clips for EXP-0010C from ground truth only (rule: protocol_v3.json -> fresh_holdout).

Candidate layouts are those NOT used by any earlier choice or diagnosis: L01, L03, L04, L05, L06, L07 (not L00, not L02/L08/L09/L10/L11).
No detector output exists for any of them. Output: results/control_semantics/fresh_selection.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import gt3d
from analysis.edge_inference.common import cache_dir, sha256_file
from . import gp_metric as GP
from . import ground
from .survey import SCENES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "control_semantics"
FRESH_LAYOUTS = ("L01", "L03", "L04", "L05", "L06", "L07")
MIN_PERSON_H, MIN_HUMANOID_FRAC, MIN_POS_RATE_B, MIN_POS_RATE_A = 60.0, 0.5, 0.01, 0.01


def main():
    survey = json.loads((OUT / "survey.json").read_text())["rows"]
    rows = []
    for r in survey:
        if r["layout"] not in FRESH_LAYOUTS:
            continue
        split = next(s for l, sc, s in SCENES if sc == r["scene"])
        gtp = cache_dir() / "raw" / split / r["scene"] / "ground_truth.json"
        calp = cache_dir() / "meta" / split / r["scene"] / "calibration.json"
        e = {k: r[k] for k in ("layout", "scene", "camera", "person_geometry_passes", "humanoid_geometry_passes", "humanoid_frame_fraction",
                               "vehicle_frame_fraction", "median_person_height_px")}
        e["gt_file_bytes"] = gtp.stat().st_size
        base = (r["person_geometry_passes"] and r["humanoid_geometry_passes"] and (r["median_person_height_px"] or 0) >= MIN_PERSON_H
                and r["humanoid_frame_fraction"] >= MIN_HUMANOID_FRAC)
        e["pre_eligible"] = bool(base)
        if base:
            g, nf, _ = gt3d.parse_gt3d(gtp, r["camera"])
            cam = gt3d.load_camera(calp, r["camera"])
            cells = ground.visible_cells(cam, GP.CELL_M, 0.5, GP.MAX_DEPTH_M)
            for pol, cls in GP.POLICIES.items():
                pos, neg, info = GP.gt_cell_states(nf, cam, cells, g, cls)
                e[f"pos_rate_{pol[0]}"] = float(pos.mean())
                e[f"agents_scored_{pol[0]}"] = info["agents_scored"]
            e["n_cells"] = len(cells)
            e["eligible"] = bool(e["pos_rate_A"] >= MIN_POS_RATE_A and e["pos_rate_B"] >= MIN_POS_RATE_B)
        else:
            e["eligible"] = False
        rows.append(e)
        print(e["layout"], e["scene"], e["camera"], "pre", e["pre_eligible"], "eligible", e["eligible"], {k: e[k] for k in e if k.startswith("pos_rate")}, flush=True)
    el = sorted([e for e in rows if e["eligible"]], key=lambda e: (e["gt_file_bytes"], e["layout"], e["camera"]))
    chosen, used = [], set()
    for e in el:
        if e["layout"] not in used and len(chosen) < 2:
            chosen.append({k: e[k] for k in ("layout", "scene", "camera")})
            used.add(e["layout"])
    proto = Path(__file__).with_name("protocol_v3.json")
    out = {"rule": "protocol_v3.json fresh_holdout.layout_rule", "protocol_sha256": sha256_file(proto), "candidates": rows, "chosen": chosen,
           "reserve_layouts_not_chosen": sorted(set(e["layout"] for e in el) - used)}
    (OUT / "fresh_selection.json").write_text(json.dumps(out, indent=1, default=float))
    print("chosen:", chosen, "reserve:", out["reserve_layouts_not_chosen"])


if __name__ == "__main__":
    sys.exit(main())
