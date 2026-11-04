"""GT-only survey of layouts/cameras: class mix, person and humanoid geometry consistency (no detector involved).

Written before protocol_v3 so that policy definitions and the fresh-holdout rule rest on measured class mix and on a
validated humanoid geometry, not on assumptions. Output: results/control_semantics/survey.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import consistency as C
from analysis.control_gate import gt3d
from analysis.edge_inference.common import cache_dir

OUT = Path(__file__).resolve().parents[2] / "results" / "control_semantics"
NAMES = {0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter", 4: "FourierGR1T2", 5: "AgilityDigit", 6: "PalletTruck"}
# (layout, scene, split). L08/L02/L10/L11 are the already-inspected (development) layouts; the rest are unused so far.
SCENES = [("L01", "Warehouse_008", "train"), ("L03", "Warehouse_011", "train"), ("L04", "Warehouse_012", "train"),
          ("L05", "Warehouse_013", "train"), ("L06", "Warehouse_014", "train"), ("L07", "Warehouse_015", "train"),
          ("L02", "Warehouse_010", "train"), ("L08", "Warehouse_016", "train"), ("L10", "Warehouse_019", "train"),
          ("L11", "Warehouse_022", "val")]
Q = [5, 25, 50, 75, 95]


def humanoid_consistency(g: dict, cam: dict, cls_ids=(4, 5)) -> dict:
    """Humanoid rows wholly inside the frame: visible height / projected 3D-box height and foot offsets (3D box bottom)."""
    m = np.nonzero(np.isin(g["cls"], cls_ids))[0]
    if len(m) == 0:
        return {"n": 0}
    s = {k: v[m] for k, v in g.items()}
    vis = np.stack([s[k] for k in ("x1", "y1", "x2", "y2")], 1)
    fz = s["cz"] - s["sz"] / 2
    foot = np.stack([s["cx"], s["cy"], fz], 1)
    top = foot.copy()
    top[:, 2] = fz + s["sz"]
    u0, v0, z0 = gt3d.project(foot, cam)
    _, v1, _ = gt3d.project(top, cam)
    eh = v0 - v1
    ins = C.interior(vis, z0) & (eh > 20)
    if ins.sum() < 50:
        return {"n": int(len(m)), "n_interior": int(ins.sum())}
    vh = vis[:, 3] - vis[:, 1]
    return {"n": int(len(m)), "n_interior": int(ins.sum()),
            "vis_h_over_3dbox_h_p5_25_50_75_95": np.percentile((vh / eh)[ins], Q).round(3).tolist(),
            "foot_v_minus_visible_bottom_px_p5_25_50_75_95": np.percentile((v0 - vis[:, 3])[ins], Q).round(1).tolist(),
            "foot_u_minus_visible_centre_px_p5_25_50_75_95": np.percentile((u0 - (vis[:, 0] + vis[:, 2]) / 2)[ins], Q).round(1).tolist(),
            "median_3dbox_height_m": float(np.median(s["sz"]))}


def humanoid_passes(c: dict) -> bool:
    """Same tolerance as the person rule: median ratio in [0.9, 1.1], median foot offsets <= 15 px."""
    if "vis_h_over_3dbox_h_p5_25_50_75_95" not in c:
        return False
    return (0.9 <= c["vis_h_over_3dbox_h_p5_25_50_75_95"][2] <= 1.1 and abs(c["foot_v_minus_visible_bottom_px_p5_25_50_75_95"][2]) <= 15
            and abs(c["foot_u_minus_visible_centre_px_p5_25_50_75_95"][2]) <= 15)


def survey_camera(layout, scene, split, camera) -> dict | None:
    gtp = cache_dir() / "raw" / split / scene / "ground_truth.json"
    calp = cache_dir() / "meta" / split / scene / "calibration.json"
    try:
        g, nf, _ = gt3d.parse_gt3d(gtp, camera)
        cam = gt3d.load_camera(calp, camera)
    except KeyError:
        return None
    if nf == 0 or len(g["frame"]) == 0:
        return None
    cons = C.scene_consistency(g, cam)
    row = {"layout": layout, "scene": scene, "camera": camera, "n_frames": nf, "person_geometry_passes": bool(C.passes(cons))}
    row["class_rows"] = {NAMES[c]: int((g["cls"] == c).sum()) for c in np.unique(g["cls"])}
    for key, ids in (("person", (0,)), ("humanoid", (4, 5)), ("vehicle", (1, 6)), ("amr_transporter", (2, 3))):
        f = np.zeros(nf, bool)
        f[g["frame"][np.isin(g["cls"], ids)]] = True
        row[f"{key}_frame_fraction"] = round(float(f.mean()), 3)
    pm = g["cls"] == 0
    row["median_person_height_px"] = float(np.median((g["y2"] - g["y1"])[pm])) if pm.any() else None
    hc = humanoid_consistency(g, cam)
    row["humanoid_consistency"] = hc
    row["humanoid_geometry_passes"] = bool(humanoid_passes(hc))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout, scene, split in SCENES:
        if not (cache_dir() / "raw" / split / scene / "ground_truth.json").exists():
            continue
        for k in range(6):
            r = survey_camera(layout, scene, split, f"Camera_{k:04d}")
            if r:
                rows.append(r)
                print(layout, scene, r["camera"], "person_geom", r["person_geometry_passes"], "humanoid_frames", r["humanoid_frame_fraction"],
                      "hum_geom", r["humanoid_geometry_passes"], "vehicle_frames", r["vehicle_frame_fraction"], "h", r["median_person_height_px"], flush=True)
    (OUT / "survey.json").write_text(json.dumps({"purpose": "GT-only class mix and geometry survey before protocol_v3", "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
