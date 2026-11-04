import json, numpy as np
from analysis.control_gate import run_cag, zones, gt3d
out = {}
for role in ("hold_L08", "hold_L02"):
    clip, _ = run_cag.load_clip(role); g = clip.g
    _, _, calp, cam = run_cag.paths(role); camera = gt3d.load_camera(calp, cam)
    h = (g["cls"] == 4) | (g["cls"] == 5)
    foot3 = np.stack([g["cx"][h], g["cy"][h], g["cz"][h] - g["sz"][h] / 2], 1)
    u, v, z = gt3d.project(foot3, camera); ok = z > 0.1
    pos, neg = zones.gt_states_tolerant(clip.n_frames, clip.foot, clip.frame, clip.vis_h, clip.depth_ok, np.stack([u, v], 1)[ok], g["frame"][h][ok])
    out[role] = {}
    for cfg in ("S0", "YT"):
        dets, n = run_cag.load_dets(role, cfg, 1); fm = np.arange(min(n, clip.n_frames)); rows = []
        for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            m = zones.occupancy_metrics(zones.occupancy_counts(pos, neg, zones.det_states(clip.n_frames, dets, t), fm))
            rows.append((t, round(m["tpr"], 3), round(m["fpr"], 3), round(m["balanced_accuracy"], 3)))
        out[role][cfg] = rows; print(role, cfg, rows, flush=True)
json.dump(out, open(str(run_cag.RESULTS / "posthoc_tolerant_tau_sweep.json"), "w"), indent=1)
