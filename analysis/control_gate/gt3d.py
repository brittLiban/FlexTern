"""Ground truth with 3D boxes: whole-person 2D boxes by projection, occlusion proxy, ground-plane foot points.

The dataset's 2D labels are *visible extents*. For every object the GT also gives a 3D box
(centre, scale = (dx, dy, dz) metres, rotation = (0, 0, yaw)) in a world frame, and each scene's
calibration.json gives per-camera intrinsic K and extrinsic [R|t] (world -> camera, y-down image).
Projecting the eight 3D corners gives a *whole-extent* 2D box; visible_area / whole_area is an
occlusion measure the dataset does not provide. The projection convention (axis order, yaw sign)
is validated empirically against the visible boxes of unoccluded persons (see validate()).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASS_IDS = {"Person": 0, "Forklift": 1, "NovaCarter": 2, "Transporter": 3,
             "FourierGR1T2": 4, "AgilityDigit": 5, "PalletTruck": 6}


def load_camera(calib_path: Path | str, camera: str) -> dict:
    c = json.load(open(calib_path, encoding="utf8"))
    for s in c["sensors"]:
        if s.get("id") == camera:
            return {"K": np.asarray(s["intrinsicMatrix"], np.float64), "E": np.asarray(s["extrinsicMatrix"], np.float64)}
    raise KeyError(camera)


def parse_gt3d(path: Path | str, camera: str):
    """Stream the scene GT; keep every object visible in `camera` with its 3D box. Returns dict of arrays."""
    import ijson
    cols = {k: [] for k in ("frame", "oid", "cls", "x1", "y1", "x2", "y2", "cx", "cy", "cz", "sx", "sy", "sz", "yaw")}
    n_present = []
    with open(path, "rb") as f:
        for fk, objs in ijson.kvitems(f, ""):
            fr = int(fk)
            n_present.append(fr)
            for o in objs:
                vis = o.get("2d bounding box visible", o.get("2d_bounding_box_visible", {}))
                if camera not in vis:
                    continue
                loc = o.get("3d location", o.get("3d_location"))
                sc = o.get("3d bounding box scale", o.get("3d_bounding_box_scale"))
                rot = o.get("3d bounding box rotation", o.get("3d_bounding_box_rotation"))
                b = vis[camera]
                cols["frame"].append(fr)
                cols["oid"].append(int(o.get("object id", o.get("object_id"))))
                cols["cls"].append(CLASS_IDS[o.get("object type", o.get("object_type"))])
                for k, v in zip(("x1", "y1", "x2", "y2"), b):
                    cols[k].append(float(v))
                for k, v in zip(("cx", "cy", "cz"), loc):
                    cols[k].append(float(v))
                for k, v in zip(("sx", "sy", "sz"), sc):
                    cols[k].append(float(v))
                cols["yaw"].append(float(rot[2]))
    out = {k: np.asarray(v, np.int32 if k in ("frame", "oid", "cls") else np.float64) for k, v in cols.items()}
    nf = int(max(n_present)) + 1 if n_present else 0
    return out, nf, len(set(n_present))


def box_corners(cx, cy, cz, sx, sy, sz, yaw, swap_xy=False, yaw_sign=1.0):
    """(N,8,3) world corners. Local box: x extent sx, y extent sy, z extent sz, rotated by yaw about z."""
    if swap_xy:
        sx, sy = sy, sx
    sgn = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64) * 0.5
    loc = sgn[None] * np.stack([sx, sy, sz], 1)[:, None, :]
    c, s = np.cos(yaw_sign * yaw), np.sin(yaw_sign * yaw)
    x = loc[:, :, 0] * c[:, None] - loc[:, :, 1] * s[:, None]
    y = loc[:, :, 0] * s[:, None] + loc[:, :, 1] * c[:, None]
    return np.stack([x + cx[:, None], y + cy[:, None], loc[:, :, 2] + cz[:, None]], axis=2)


def project(points: np.ndarray, cam: dict):
    """points (...,3) world -> (u, v, depth)."""
    P = points.reshape(-1, 3)
    pc = P @ cam["E"][:, :3].T + cam["E"][:, 3]
    z = pc[:, 2]
    uv = pc @ cam["K"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    return u.reshape(points.shape[:-1]), v.reshape(points.shape[:-1]), z.reshape(points.shape[:-1])


def whole_boxes(g: dict, cam: dict, swap_xy=False, yaw_sign=1.0, W=1920, H=1080):
    """Whole-extent 2D box (unclipped), foot point (bottom-centre of the 3D box, projected), depth ok flag."""
    corners = box_corners(g["cx"], g["cy"], g["cz"], g["sx"], g["sy"], g["sz"], g["yaw"], swap_xy, yaw_sign)
    u, v, z = project(corners, cam)
    ok = (z > 0.1).all(axis=1)
    box = np.stack([u.min(1), v.min(1), u.max(1), v.max(1)], 1)
    foot3 = np.stack([g["cx"], g["cy"], g["cz"] - g["sz"] / 2], 1)
    fu, fv, fz = project(foot3, cam)
    return box, np.stack([fu, fv], 1), ok & (fz > 0.1)


def _iou_pairs(a, b):
    ix1, iy1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix2, iy2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def visible_fraction(vis_box: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """Area(visible)/Area(whole clipped to the image) -- 1.0 means nothing hides this person (or truncation by frame edge)."""
    w = whole.copy()
    w[:, [0, 2]] = np.clip(w[:, [0, 2]], 0, 1920)
    w[:, [1, 3]] = np.clip(w[:, [1, 3]], 0, 1080)
    av = np.clip(vis_box[:, 2] - vis_box[:, 0], 0, None) * np.clip(vis_box[:, 3] - vis_box[:, 1], 0, None)
    aw = np.clip(w[:, 2] - w[:, 0], 0, None) * np.clip(w[:, 3] - w[:, 1], 0, None)
    return np.where(aw > 0, av / np.maximum(aw, 1e-9), 0.0)


def validate(g: dict, cam: dict, cls_id: int = 0, n_max: int = 20000) -> list[dict]:
    """Try the four axis/yaw conventions; report agreement of projected-vs-visible boxes for persons."""
    m = np.nonzero(g["cls"] == cls_id)[0][:n_max]
    sub = {k: v[m] for k, v in g.items()}
    vis = np.stack([sub["x1"], sub["y1"], sub["x2"], sub["y2"]], 1)
    rows = []
    for swap in (False, True):
        for ys in (1.0, -1.0):
            box, foot, ok = whole_boxes(sub, cam, swap, ys)
            iou = _iou_pairs(vis, box)
            inside = (vis[:, 0] > 2) & (vis[:, 1] > 2) & (vis[:, 2] < 1918) & (vis[:, 3] < 1078) & ok
            rows.append({"swap_xy": swap, "yaw_sign": ys, "median_iou_all": float(np.median(iou[ok])) if ok.any() else None,
                         "p90_iou_all": float(np.percentile(iou[ok], 90)) if ok.any() else None,
                         "frac_iou_ge_0.8_inside": float((iou[inside] >= 0.8).mean()) if inside.any() else None})
    return rows
