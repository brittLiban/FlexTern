"""CLI for EXP-0010C: lock, fresh-holdout fetch/detect/score, short timing comparison.

  lock                       write results/control_semantics/lock.json (taus from development, chosen fresh clips, hashes)
  fetch                      download the two fresh videos (refuses unless the lock is committed and code is clean)
  detect CONFIG ROLE         offline detections on a fresh clip (9000 frames); manifest records the whole-repository state
  score ROLE                 ground-plane metric for both policies + the EXP-0010B v2 metric as written, ONE scoring per clip
  replay CONFIG ROLE ...     paced 30 fps replay for the timing comparison
Fresh roles are named hold_<layout> (hold_L05, hold_L03) so the EXP-0010B machinery (paths, caches, replay) is reused unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis.control_gate import cag_eval, gt3d, run_cag
from analysis.edge_inference.common import cache_dir, sha256_file
from . import gp_metric as GP
from . import ground, provenance

REPO = provenance.REPO
PKG = Path(__file__).resolve().parent
RES = REPO / "results" / "control_semantics"
PROTOCOL = PKG / "protocol_v3.json"
LOCK = RES / "lock.json"
V2_TAU = 0.05              # EXP-0010B frozen YOLOX-Tiny tau for the as-written v2 metric


def fresh_clips() -> dict:
    sel = json.loads((RES / "fresh_selection.json").read_text())["chosen"]
    split = "train"
    return {f"hold_{c['layout']}": (split, c["scene"], c["camera"]) for c in sel}


def register_roles():
    """Make the fresh roles known to the EXP-0010B machinery and redirect its outputs to results/control_semantics."""
    if LOCK.exists() or (RES / "fresh_selection.json").exists():
        for role in [r for r in list(run_cag.CLIPS) if r.startswith("hold_")]:
            del run_cag.CLIPS[role]       # EXP-0010B holdout roles (L08, L02) are development clips here; keep them addressable under new names
        run_cag.CLIPS.update(fresh_clips())
    run_cag.RESULTS = RES
    orig_manifest = run_cag.manifest

    def manifest(role, config, mode, threads, extra=None):
        m = orig_manifest(role, config, mode, threads, extra)
        st = provenance.repo_state()
        m["repo_state"] = st
        m["environment"]["git"] = {"commit": st["commit"], "dirty": st["dirty"], "scope": st["scope"],
                                   "note": "whole-repository state (EXP-0010C provenance); the Stage A git_info field is not used"}
        m["protocol_v3_sha256"] = sha256_file(PROTOCOL)
        return m

    run_cag.manifest = manifest
    run_cag.check_lock = require_lock


def require_lock(role=None) -> dict:
    """Fresh-holdout guard: lock committed + matches protocol + code clean."""
    if not LOCK.exists():
        sys.exit("fresh holdout is locked: results/control_semantics/lock.json does not exist")
    lock = json.loads(LOCK.read_text())
    if lock["protocol_v3_sha256"] != sha256_file(PROTOCOL):
        sys.exit("protocol_v3.json changed after the lock")
    tracked = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", str(REPO), "ls-files", "--error-unmatch", str(LOCK.relative_to(REPO))], capture_output=True, text=True)
    if tracked.returncode != 0:
        sys.exit("lock.json is not committed; commit it before touching any fresh clip")
    return provenance.require_clean_code()


def cmd_lock(a):
    dev = json.loads((RES / "dev_policy_scores.json").read_text())
    sel = json.loads((RES / "fresh_selection.json").read_text())
    lock = {"protocol_v3_sha256": sha256_file(PROTOCOL),
            "tau": {pol: dev["policies"][pol]["best_tau"] for pol in dev["policies"]},
            "tau_v2_as_written": V2_TAU, "deployment_threads": 8,
            "fresh_holdout": sel["chosen"], "reserve_layouts_unused": sel["reserve_layouts_not_chosen"],
            "hashes": {f: sha256_file(RES / f) for f in ("dev_policy_scores.json", "fresh_selection.json", "localization.json", "survey.json",
                                                           "fa_attribution.json", "miss_attribution.json")},
            "repo_state_at_lock": provenance.repo_state()}
    LOCK.write_text(json.dumps(lock, indent=1))
    print(json.dumps({k: lock[k] for k in ("tau", "fresh_holdout", "reserve_layouts_unused")}, indent=1))


def cmd_fetch(a):
    require_lock()
    from analysis.edge_inference.dataset_audit import download
    for role, (split, scene, cam) in fresh_clips().items():
        p = download(split, scene, f"videos/{cam}.mp4")
        print(role, p, p.stat().st_size, flush=True)


def cmd_detect(a):
    require_lock()
    run_cag.cmd_detect(a)


def cmd_replay(a):
    require_lock()
    run_cag.cmd_replay(a)


def _policy_eval(role, dets, n, lock):
    clip, _ = run_cag.load_clip(role)
    _, _, calp, cam_id = run_cag.paths(role)
    cam = gt3d.load_camera(calp, cam_id)
    frames = np.arange(min(n, clip.n_frames))
    cells = ground.visible_cells(cam, GP.CELL_M, 0.5, GP.MAX_DEPTH_M)
    out = {"role": role, "n_cells": len(cells), "frames": int(len(frames)), "policies": {}}
    for pol, cls in GP.POLICIES.items():
        tau = lock["tau"][pol]
        pos, neg, info = GP.gt_cell_states(clip.n_frames, cam, cells, clip.g, cls)
        pred, dinfo = GP.det_cell_states(clip.n_frames, cam, cells, dets, tau)
        m = GP.metrics_for(pos, neg, pred, frames)
        bt = GP.block_bootstrap(pos, neg, pred, frames)
        ok = bool(m["tpr"] >= 0.90 and bt["tpr_ci"][0] >= 0.85 and m["fa_per_predicted"] <= 0.10 and bt["precision_cell_ci"][0] >= 0.85)
        out["policies"][pol] = {"tau": tau, "metrics": {k: m[k] for k in ("tpr", "fpr", "balanced_accuracy", "precision_cell", "fa_per_predicted",
                                                                          "frame_fa_rate", "tp", "fn", "fp", "tn", "n_pos", "n_neg")},
                                "bootstrap": bt, "gt": info, "detections": dinfo, "quality_bar_met": ok,
                                "pos_rate": float(pos[frames].mean())}
        print(role, pol, "tau", tau, "TPR %.3f FA/pred %.3f FPR %.3f BA %.3f" % (m["tpr"], m["fa_per_predicted"], m["fpr"], m["balanced_accuracy"]), "bar", ok, flush=True)
    return out, clip, dets


def cmd_score(a):
    st = require_lock()
    lock = json.loads(LOCK.read_text())
    target = RES / f"fresh_{a.role}_{a.config}.json"
    if target.exists():
        sys.exit(f"{target.name} already exists: the fresh holdout is scored ONCE per clip and configuration")
    dets, n = run_cag.load_dets(a.role, a.config, 1)
    res, clip, dets = _policy_eval(a.role, dets, n, lock)
    res["repo_state"] = st
    res["protocol_v3_sha256"] = sha256_file(PROTOCOL)
    res["lock_sha256"] = sha256_file(LOCK)
    frames = np.arange(min(n, clip.n_frames))
    v2 = cag_eval.evaluate(clip, dets, lock["tau_v2_as_written"], frames, boot=True)
    z = v2["zone"]
    res["v2_image_grid_as_written"] = v2
    if z:
        print(a.role, "v2 as written (image grid, tau %.2f): BA %.3f TPR %.3f FPR %.3f" % (lock["tau_v2_as_written"], z["balanced_accuracy"], z["tpr"], z["fpr"]), flush=True)
    target.write_text(json.dumps(res, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    sub.add_parser("fetch")
    p = sub.add_parser("detect"); p.add_argument("config"); p.add_argument("role"); p.add_argument("--stride", type=int, default=1)
    p.add_argument("--threads", type=int, default=8); p.add_argument("--max-frames", type=int, default=None)
    p = sub.add_parser("score"); p.add_argument("role"); p.add_argument("--config", default="YT")
    p = sub.add_parser("replay"); p.add_argument("config"); p.add_argument("role"); p.add_argument("--threads", type=int, default=8)
    p.add_argument("--rep", type=int, default=0); p.add_argument("--max-frames", type=int, default=None); p.add_argument("--tag", default="t")
    a = ap.parse_args(argv)
    register_roles()
    {"lock": cmd_lock, "fetch": cmd_fetch, "detect": cmd_detect, "score": cmd_score, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    main()
