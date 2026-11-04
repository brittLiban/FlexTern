"""Choose the fresh holdout (layout, camera) clips from ground truth alone (rule: protocol_v2.json -> fresh_holdout).

No detector output exists for these clips when this runs. Output: results/control_gate/holdout_selection.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.edge_inference.common import RESULTS as _R, cache_dir, sha256_file  # noqa: F401
from . import consistency as C
from . import gt3d, zones

REPO_RESULTS = Path(__file__).resolve().parents[2] / "results" / "control_gate"
CANDIDATE_SCENES = {"L01": "Warehouse_008", "L02": "Warehouse_010", "L03": "Warehouse_011", "L04": "Warehouse_012",
                    "L05": "Warehouse_013", "L06": "Warehouse_014", "L07": "Warehouse_015", "L08": "Warehouse_016"}


def zone_rates(g, cam, n_frames):
    m, vis, foot, eh, dz = C.person_geometry(g, cam, "box_bottom")
    height = vis[:, 3] - vis[:, 1]
    pos, _ = zones.gt_states(n_frames, foot, g["frame"][m], height, dz > 0.1)
    return pos.mean(0)


def evaluate_clip(scene: str, split: str, camera: str) -> dict:
    gtp = cache_dir() / "raw" / split / scene / "ground_truth.json"
    calp = cache_dir() / "meta" / split / scene / "calibration.json"
    g, nf, _ = gt3d.parse_gt3d(gtp, camera)
    cam = gt3d.load_camera(calp, camera)
    cons = C.scene_consistency(g, cam)
    ok = C.passes(cons)
    pm = g["cls"] == 0
    h = (g["y2"] - g["y1"])[pm]
    veh = np.zeros(nf, bool)
    vm = (g["cls"] == 1) | (g["cls"] == 6)
    veh[g["frame"][vm]] = True
    rates = zone_rates(g, cam, nf) if ok else np.zeros(9)
    nontrivial = int(((rates >= 0.10) & (rates <= 0.90)).sum())
    return {"scene": scene, "camera": camera, "n_frames": nf, "geometry_passes": bool(ok), "consistency": cons,
            "median_person_height_px": float(np.median(h)) if len(h) else None,
            "vehicle_frame_fraction": float(veh.mean()), "zone_pos_rates": rates.round(3).tolist(),
            "nontrivial_zones": nontrivial, "gt_file_bytes": gtp.stat().st_size,
            "eligible": bool(ok and len(h) and np.median(h) >= 60 and veh.mean() >= 0.20 and nontrivial >= 3)}


def main():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout, scene in CANDIDATE_SCENES.items():
        gtp = cache_dir() / "raw" / "train" / scene / "ground_truth.json"
        if not gtp.exists():
            rows.append({"layout": layout, "scene": scene, "status": "gt not downloaded"})
            continue
        for camera in ("Camera_0000", "Camera_0001", "Camera_0002", "Camera_0003", "Camera_0004", "Camera_0005"):
            try:
                r = evaluate_clip(scene, "train", camera)
            except (KeyError, StopIteration):
                continue
            if r["n_frames"] == 0 or r["median_person_height_px"] is None:
                continue
            r["layout"] = layout
            rows.append(r)
            print(layout, scene, camera, "geom", r["geometry_passes"], "h", r["median_person_height_px"], "veh", round(r["vehicle_frame_fraction"], 2),
                  "nontriv", r["nontrivial_zones"], "eligible", r["eligible"], flush=True)
    el = [r for r in rows if r.get("eligible")]
    el.sort(key=lambda r: (-r["nontrivial_zones"], r["gt_file_bytes"]))
    chosen, used = [], set()
    for r in el:
        if r["layout"] not in used and len(chosen) < 2:
            chosen.append({k: r[k] for k in ("layout", "scene", "camera")})
            used.add(r["layout"])
    out = {"rule": "protocol_v2.json fresh_holdout.layout_rule", "protocol_sha256": sha256_file(Path(__file__).with_name("protocol_v2.json")),
           "candidates": rows, "chosen": chosen}
    (REPO_RESULTS / "holdout_selection.json").write_text(json.dumps(out, indent=1, default=float))
    print("chosen:", chosen)


if __name__ == "__main__":
    sys.exit(main())
