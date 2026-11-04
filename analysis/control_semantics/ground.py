"""Ground-plane geometry from the scene calibration (numpy only).

World frame: metres, z up. extrinsic E = [R|t] maps world -> camera, K is the intrinsic matrix (image u right, v down).
`pixel_to_ground` intersects the viewing ray of a pixel with the plane z = z0 (the floor, z0 = 0 unless a clip says
otherwise). `ground_to_pixel` is the forward projection of (x, y, z0). Both use the same camera dictionaries as
`analysis.control_gate.gt3d.load_camera`.
"""
from __future__ import annotations

import numpy as np

W, H = 1920, 1080


def camera_centre(cam: dict) -> np.ndarray:
    R, t = cam["E"][:, :3], cam["E"][:, 3]
    return -R.T @ t


def pixel_to_ground(uv: np.ndarray, cam: dict, z0: float = 0.0):
    """(N,2) pixels -> ((N,2) ground x,y in metres, (N,) camera-frame depth, (N,) bool ok). ok=False: ray misses the plane in front."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    R = cam["E"][:, :3]
    C = camera_centre(cam)
    rays_cam = np.linalg.inv(cam["K"]) @ np.vstack([uv.T, np.ones(len(uv))])      # (3,N)
    d = R.T @ rays_cam                                                          # world directions
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z0 - C[2]) / d[2]
    pts = C[:, None] + d * s
    depth = (R @ (pts - C[:, None]))[2]                                         # camera z of the hit point
    ok = np.isfinite(s) & (s > 0) & (depth > 0.1)
    return pts[:2].T, depth, ok


def ground_to_pixel(xy: np.ndarray, cam: dict, z0: float = 0.0):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    P = np.column_stack([xy, np.full(len(xy), z0)])
    pc = P @ cam["E"][:, :3].T + cam["E"][:, 3]
    uv = pc @ cam["K"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]]), pc[:, 2]


def metres_per_pixel_vertical(cam: dict, v: float, u: float = W / 2, z0: float = 0.0) -> float:
    """Ground distance (m) covered by one image row at pixel (u, v): sensitivity of ground position to foot-row error."""
    g, _, ok = pixel_to_ground(np.array([[u, v], [u, v + 1.0]]), cam, z0)
    return float(np.linalg.norm(g[1] - g[0])) if ok.all() else float("nan")


def cell_index(xy: np.ndarray, cell_m: float) -> np.ndarray:
    """(N,2) metres -> (N,2) integer cell indices of a square ground grid anchored at the world origin."""
    return np.floor(np.asarray(xy, np.float64) / cell_m).astype(np.int64)


def visible_cells(cam: dict, cell_m: float, min_frac: float = 0.5, max_depth_m: float = 60.0, z0: float = 0.0, n: int = 5):
    """Ground cells whose area is at least `min_frac` inside the image (n x n sample points per cell) and in front of the
    camera within `max_depth_m`. Returns a sorted list of (ix, iy). Derived from calibration only (no detector, no labels)."""
    # coarse search: back-project a dense pixel lattice to find the ground extent, then test every candidate cell
    us, vs = np.meshgrid(np.linspace(0, W - 1, 48), np.linspace(0, H - 1, 27))
    g, depth, ok = pixel_to_ground(np.column_stack([us.ravel(), vs.ravel()]), cam, z0)
    g = g[ok & (depth < max_depth_m)]
    if len(g) == 0:
        return []
    lo, hi = np.floor(g.min(0) / cell_m).astype(int) - 1, np.floor(g.max(0) / cell_m).astype(int) + 1
    out = []
    frac = (np.arange(n) + 0.5) / n
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            pts = np.array([[(ix + a) * cell_m, (iy + b) * cell_m] for a in frac for b in frac])
            uv, z = ground_to_pixel(pts, cam, z0)
            inside = (z > 0.1) & (z < max_depth_m * 1.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.mean() >= min_frac:
                out.append((ix, iy))
    return out
