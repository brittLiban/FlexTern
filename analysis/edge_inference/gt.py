"""Ground-truth handling for MTMC_Tracking_2026 scenes (per-camera 2D boxes).

Real key names in ground_truth.json contain spaces ("object type", "object id",
"2d bounding box visible"), unlike the underscore names in the dataset README; both
are accepted so this does not break if NVIDIA fixes the files.

The raw JSON is 15-280 MB per scene (all cameras, 3D boxes included). It is stream-parsed
with ijson and reduced to one compact .npz per camera that holds only what Stage A scores.
Frames with no annotation record are *absent from the JSON*; whether that means "no
objects" or "unlabeled" is an assumption that is recorded in the manifest, not hidden:
the JSON has a record for every frame index 0..N-1 (empty lists for empty frames), so
absent == unlabeled and is counted separately by `frame_coverage`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLASS_IDS = {"Person": 0, "Forklift": 1, "NovaCarter": 2, "Transporter": 3,
             "FourierGR1T2": 4, "AgilityDigit": 5, "PalletTruck": 6}
HUMANOID_CLASSES = ("FourierGR1T2", "AgilityDigit")


def _k(d: dict, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(names)


def _num(x):
    return float(x) if not isinstance(x, (int, float)) else x


def parse_ground_truth(path: Path | str, cameras: set[str] | None = None):
    """Stream the scene GT. Returns (per_camera_columns, n_frames, frames_present).

    per_camera_columns[cam] = dict of equal-length np arrays: frame, x1,y1,x2,y2, cls, oid
    """
    import ijson  # lazy: host tests that don't parse real files need no ijson

    cols: dict[str, dict[str, list]] = {}
    frames_present = []
    with open(path, "rb") as f:
        for frame_key, objs in ijson.kvitems(f, ""):
            fr = int(frame_key)
            frames_present.append(fr)
            for o in objs:
                cname = _k(o, "object type", "object_type")
                vis = _k(o, "2d bounding box visible", "2d_bounding_box_visible")
                for cam, box in vis.items():
                    if cameras is not None and cam not in cameras:
                        continue
                    c = cols.setdefault(cam, {k: [] for k in ("frame", "x1", "y1", "x2", "y2", "cls", "oid")})
                    c["frame"].append(fr)
                    for k, v in zip(("x1", "y1", "x2", "y2"), box):
                        c[k].append(float(_num(v)))
                    c["cls"].append(CLASS_IDS[cname])
                    c["oid"].append(int(_k(o, "object id", "object_id")))
    out = {cam: {k: np.asarray(v, dtype=np.int32 if k in ("frame", "cls", "oid") else np.float32)
                 for k, v in c.items()} for cam, c in cols.items()}
    fp = np.asarray(sorted(frames_present), dtype=np.int32)
    n_frames = int(fp.max()) + 1 if len(fp) else 0
    return out, n_frames, fp


def save_camera_gt(path: Path, cols: dict, n_frames: int, frames_present: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n_frames=np.int32(n_frames), frames_present=frames_present, **cols)


def load_camera_gt(path: Path | str) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["n_frames"] = int(d["n_frames"])
    return d


def frame_coverage(frames_present: np.ndarray, n_frames: int) -> dict:
    """How many frame indices have an annotation record (empty or not)."""
    present = int(len(np.unique(frames_present)))
    return {"n_frames": int(n_frames), "frames_with_record": present,
            "frames_without_record": int(n_frames) - present}


def size_stats(gt: dict, cls_id: int = 0) -> dict:
    """Person-size distribution at native resolution (ground truth only, no model output)."""
    m = gt["cls"] == cls_id
    h = (gt["y2"] - gt["y1"])[m]
    w = (gt["x2"] - gt["x1"])[m]
    if len(h) == 0:
        return {"n": 0}
    q = np.percentile(h, [5, 25, 50, 75, 95]).round(1).tolist()
    return {"n": int(m.sum()), "frames_with_person": int(len(np.unique(gt["frame"][m]))),
            "height_px_p5_25_50_75_95": q, "median_width_px": float(np.median(w)),
            "frac_h_ge_32": float((h >= 32).mean()), "frac_h_ge_64": float((h >= 64).mean()),
            "frac_h_ge_128": float((h >= 128).mean())}


def content_sha256(gt: dict) -> str:
    """Hash of the array *contents* (npz container bytes are not reproducible)."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(gt):
        v = gt[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()
