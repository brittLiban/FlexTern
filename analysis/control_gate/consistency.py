"""GT-only calibration/3D consistency statistics (no detector involved).

For persons whose visible box is fully inside the frame, compare the visible box with the projection of a
standard-height person standing at the 3D box's foot point. Scenes where this projection reproduces the
visible boxes can supply occlusion/foot-point labels; scenes where it does not are flagged unreliable.
"""
from __future__ import annotations

import numpy as np

from . import gt3d

STD_HEIGHT_M = 1.75
Q = [5, 25, 50, 75, 95]


def person_geometry(g: dict, cam: dict, foot_mode: str = "box_bottom"):
    """Per-person-row arrays: visible box, ground/foot image point, expected whole-person height (px), depth."""
    m = np.nonzero(g["cls"] == 0)[0]
    s = {k: v[m] for k, v in g.items()}
    vis = np.stack([s[k] for k in ("x1", "y1", "x2", "y2")], 1)
    fz = s["cz"] - s["sz"] / 2 if foot_mode == "box_bottom" else np.zeros(len(m))
    foot = np.stack([s["cx"], s["cy"], fz], 1)
    u0, v0, z0 = gt3d.project(foot, cam)
    top = foot.copy()
    top[:, 2] = fz + STD_HEIGHT_M
    _, v1, _ = gt3d.project(top, cam)
    return m, vis, np.stack([u0, v0], 1), v0 - v1, z0


def interior(vis, depth):
    return (vis[:, 0] > 5) & (vis[:, 1] > 5) & (vis[:, 2] < 1915) & (vis[:, 3] < 1075) & (depth > 0.5)


def scene_consistency(g: dict, cam: dict) -> dict:
    out = {}
    for mode in ("box_bottom", "z0"):
        m, vis, foot, eh, dz = person_geometry(g, cam, mode)
        ins = interior(vis, dz) & (eh > 20)
        if ins.sum() < 50:
            out[mode] = {"n": int(ins.sum())}
            continue
        vh = vis[:, 3] - vis[:, 1]
        out[mode] = {"n_interior": int(ins.sum()),
                     "vis_h_over_expected_h_p5_25_50_75_95": np.percentile((vh / eh)[ins], Q).round(3).tolist(),
                     "foot_v_minus_visible_bottom_px_p5_25_50_75_95": np.percentile((foot[:, 1] - vis[:, 3])[ins], Q).round(1).tolist(),
                     "foot_u_minus_visible_centre_px_p5_25_50_75_95": np.percentile((foot[:, 0] - (vis[:, 0] + vis[:, 2]) / 2)[ins], Q).round(1).tolist()}
    return out


def passes(c: dict, mode: str = "box_bottom") -> bool:
    """Preregistered reliability rule for using projected geometry in a scene: median ratio in [0.9, 1.1]
    and the median foot-point error under 15 px (|dv| and |du|)."""
    r = c.get(mode, {})
    if "vis_h_over_expected_h_p5_25_50_75_95" not in r:
        return False
    med = r["vis_h_over_expected_h_p5_25_50_75_95"][2]
    dv = abs(r["foot_v_minus_visible_bottom_px_p5_25_50_75_95"][2])
    du = abs(r["foot_u_minus_visible_centre_px_p5_25_50_75_95"][2])
    return 0.9 <= med <= 1.1 and dv <= 15 and du <= 15
