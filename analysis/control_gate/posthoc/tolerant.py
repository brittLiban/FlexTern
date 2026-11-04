import json, numpy as np
from analysis.control_gate import run_cag, zones, gt3d
from analysis.edge_inference import evaluate as ev
sel = json.load(open(str(run_cag.RESULTS / "dev_selection.json")))
out = {}
for role in ("hold_L08", "hold_L02", "dev_L11", "dev_L10"):
    clip, _ = run_cag.load_clip(role); g = clip.g
    _, _, calp, cam = run_cag.paths(role)
    camera = gt3d.load_camera(calp, cam)
    stride = 1 if role.startswith("hold") else 3
    # ignore feet: humanoids (3D foot projection) ; person rows go through gt_states_tolerant (all usable feet count)
    h = (g["cls"] == 4) | (g["cls"] == 5)
    ihf = ihfr = None
    if h.any():
        foot3 = np.stack([g["cx"][h], g["cy"][h], g["cz"][h] - g["sz"][h] / 2], 1)
        u, v, z = gt3d.project(foot3, camera)
        ok = z > 0.1
        ihf = np.stack([u, v], 1)[ok]; ihfr = g["frame"][h][ok]
    out[role] = {"humanoid_rows": int(h.sum()), "configs": {}}
    pos0, neg0 = zones.gt_states(clip.n_frames, clip.foot, clip.frame, clip.vis_h, clip.depth_ok)
    pos1, neg1 = zones.gt_states_tolerant(clip.n_frames, clip.foot, clip.frame, clip.vis_h, clip.depth_ok, ihf, ihfr, neg_margin=zones.MARGIN)
    pos2, neg2 = zones.gt_states_tolerant(clip.n_frames, clip.foot, clip.frame, clip.vis_h, clip.depth_ok, ihf, ihfr)
    for cfg in ("S0", "YT"):
        if not run_cag.dets_path(role, cfg, stride).exists(): continue
        dets, n = run_cag.load_dets(role, cfg, stride)
        fm = np.arange(0, min(n, clip.n_frames), stride) if stride > 1 else np.arange(min(n, clip.n_frames))
        tau = sel["tau"].get(cfg, {"S0": 0.1, "YT": 0.05}[cfg])
        pred = zones.det_states(clip.n_frames, dets, tau)
        rows = {}
        for name, (p, ng) in {"preregistered": (pos0, neg0), "ignore_aware_same_margin": (pos1, neg1), "ignore_aware_2x_neg_margin": (pos2, neg2)}.items():
            m = zones.occupancy_metrics(zones.occupancy_counts(p, ng, pred, fm))
            bt = zones.block_bootstrap_occupancy(p[fm], ng[fm], pred[fm])
            rows[name] = {"tpr": m["tpr"], "fpr": m["fpr"], "ba": m["balanced_accuracy"], "ba_ci": bt["balanced_accuracy_ci"], "n_neg": m["n_neg"]}
        out[role]["configs"][cfg] = {"tau": tau, **rows}
        print(role, cfg, {k: (round(v["tpr"], 3), round(v["fpr"], 3), round(v["ba"], 3)) for k, v in rows.items()}, flush=True)
    if role.startswith("hold"):   # what are the 'matches a labelled Person' false alarms?
        dets, n = run_cag.load_dets(role, "YT", 1)
        k = (dets["score"] >= 0.5) & ((dets["box"][:, 3] - dets["box"][:, 1]) >= 32)
        f, b = dets["frame"][k], dets["box"][k]
        foot = np.stack([(b[:, 0] + b[:, 2]) / 2, np.minimum(b[:, 3], 1079)], 1); zm = zones.zone_masks(foot, 32.0)
        rows_by_f = {}
        for i, fr in enumerate(g["frame"]): rows_by_f.setdefault(int(fr), []).append(i)
        prow_of = {int(r): j for j, r in enumerate(clip.prow)}
        cats = {}
        for i in range(len(f)):
            if not zm[i].any() or not all(neg0[f[i], z] for z in np.nonzero(zm[i])[0]): continue
            rows = rows_by_f.get(int(f[i]), [])
            if not rows: continue
            gb = np.stack([[g["x1"][r], g["y1"][r], g["x2"][r], g["y2"][r]] for r in rows]); iou = ev.box_iou(b[i:i+1], gb)[0]
            j = int(np.argmax(iou)); r = rows[j]
            if iou[j] < 0.3 or g["cls"][r] != 0: continue
            pj = prow_of[r]
            fx, fy = clip.foot[pj]
            why = "small(<32px)" if clip.vis_h[pj] < 32 else "foot_outside_frame" if not (0 <= fx < 1920 and 0 <= fy < 1080) else "bad_depth" if not clip.depth_ok[pj] else "scored_person_foot_mismatch"
            cats[why] = cats.get(why, 0) + 1
        out[role]["matched_person_false_alarms_by_reason"] = cats
        print(role, "person-matched FA reasons", cats, flush=True)
json.dump(out, open(str(run_cag.RESULTS / "posthoc_tolerant_metric.json"), "w"), indent=1, default=float)
