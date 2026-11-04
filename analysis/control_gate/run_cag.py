"""CLI for the FT-0010 control-adequacy gate.

  detect CONFIG ROLE [--stride 3] [--threads 4]     offline detections (every STRIDE-th frame processed)
  devsel                                            development scoring, per-candidate tau, pooled selection table
  (replay / timing / holdout commands are added below)

ROLE: dev_L09 dev_L10 dev_L11 (development) | hold_L08 hold_L02 (fresh holdout, locked until dev_selection.json exists)
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis.edge_inference.common import cache_dir, environment, sha256_file
from analysis.edge_inference.replay import (ResourceSampler, _dets_to_arrays, _new_trace, _open, count_packets, frame_index,
                                            run_replay, summarize_trace, validate_trace, warmup)

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "control_gate"
PROTOCOL = PKG / "protocol_v2.json"

CLIPS = {
    "dev_L09": ("train", "Warehouse_017", "Camera_0001"),
    "dev_L10": ("train", "Warehouse_019", "Camera_0002"),
    "dev_L11": ("val", "Warehouse_022", "Camera_0002"),
    "hold_L08": ("train", "Warehouse_016", "Camera_0002"),
    "hold_L02": ("train", "Warehouse_010", "Camera_0002"),
}
HOLDOUT_FRAMES = 9000


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def paths(role: str):
    split, scene, cam = CLIPS[role]
    c = cache_dir()
    return (c / "raw" / split / scene / "videos" / f"{cam}.mp4", c / "raw" / split / scene / "ground_truth.json",
            c / "meta" / split / scene / "calibration.json", cam)


def dets_path(role: str, config: str, stride: int) -> Path:
    return cache_dir() / "cag" / "dets" / role / f"{config}_s{stride}.npz"


def run_dir(role: str, config: str, tag: str) -> Path:
    return RESULTS / "runs" / role / config / tag


def check_lock(role: str):
    if role.startswith("hold_") and not (RESULTS / "dev_selection.json").exists():
        sys.exit(f"{role} is a fresh holdout: locked until {RESULTS / 'dev_selection.json'} exists (protocol_v2.json fresh_holdout.lock)")


def load_clip(role: str):
    from . import cag_eval, gt3d
    video, gtp, calp, cam = paths(role)
    key = cache_dir() / "cag" / f"gt_{role}.npz"
    if key.exists():
        z = np.load(key)
        g = {k: z[k] for k in z.files if k != "n_frames"}
        nf = int(z["n_frames"])
    else:
        g, nf, _ = gt3d.parse_gt3d(gtp, cam)
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(key, n_frames=np.int32(nf), **g)
    return cag_eval.build_clip(role, g, nf, gt3d.load_camera(calp, cam)), video


def host_state():
    from analysis.edge_inference.run_stage_a import host_state as hs
    return hs()


def manifest(role, config, mode, threads, extra=None):
    from . import detectors as D
    video, gtp, calp, cam = paths(role)
    m = {"started_utc": utcnow(), "role": role, "config": config, "mode": mode, "threads": threads,
         "command": " ".join(["python", "-m", "analysis.control_gate.run_cag"] + sys.argv[1:]),
         "protocol_sha256": sha256_file(PROTOCOL), "protocol_id": json.loads(PROTOCOL.read_text())["protocol_id"],
         "video_sha256": sha256_file(video), "gt_json_sha256": sha256_file(gtp) if gtp.exists() else None,
         "model_files": {p.name: sha256_file(p) for p in D.model_files(config)},
         "environment": environment(), "host": host_state()}
    if extra:
        m.update(extra)
    return m


# ------------------------------------------------------------------ detect
def offline_strided(video_bytes: bytes, engine, stride: int, max_frames: int | None):
    """Decode every frame (so timing of decode is real), run the detector on frames with index % stride == 0."""
    n, tb, fps, first, W, H = count_packets(video_bytes)
    if max_frames:
        n = min(n, max_frames)
    pc = time.perf_counter_ns
    c, s = _open(video_bytes)
    dets, rows = {}, []
    i = 0
    with ResourceSampler() as rs:
        for pkt in c.demux(s):
            if pkt.pts is None:
                continue
            t0 = pc()
            for fr in pkt.decode():
                if i >= n:
                    break
                idx = frame_index(fr.pts, tb, fps, first)
                if idx % stride == 0:
                    rgb = fr.to_ndarray(format="rgb24")
                    t1 = pc()
                    x = engine.pre(rgb)
                    t2 = pc()
                    out = engine.infer(x)
                    t3 = pc()
                    b, sc = engine.post(out, W, H)
                    t4 = pc()
                    dets[idx] = (b, sc)
                    rows.append((idx, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t4 - t3) / 1e6))
                i += 1
            if i >= n:
                break
    c.close()
    return {"dets": _dets_to_arrays(dets), "rows": np.asarray(rows), "n_frames": i, "resources": rs.report(), "W": W, "H": H}


def cmd_detect(a):
    from . import detectors as D
    check_lock(a.role)
    video, _, _, _ = paths(a.role)
    data = video.read_bytes()
    engine, runner = D.build_engine(a.config, a.threads)
    warmup(engine, _first_frame(data), 10)
    max_frames = HOLDOUT_FRAMES if a.role.startswith("hold_") else a.max_frames
    r = offline_strided(data, engine, a.stride, max_frames)
    p = dets_path(a.role, a.config, a.stride)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **r["dets"], n_frames=np.int32(r["n_frames"]))
    rows = r["rows"]
    stage = {k: float(rows[:, j].mean()) for j, k in enumerate(("decode_ms", "pre_ms", "infer_ms", "post_ms"), start=1)}
    d = run_dir(a.role, a.config, f"offline_s{a.stride}")
    d.mkdir(parents=True, exist_ok=True)
    man = manifest(a.role, a.config, "offline", a.threads, {"stride": a.stride, "providers": runner.providers, "max_frames": max_frames})
    man["finished_utc"] = utcnow()
    man["detections_file"] = str(p.name)
    man["detections_sha256"] = sha256_file(p)
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    (d / "stage.json").write_text(json.dumps({"n_processed": int(len(rows)), "n_frames": r["n_frames"], "stage_ms_mean": stage,
                                              "pipeline_ms_mean_ex_decode": stage["pre_ms"] + stage["infer_ms"] + stage["post_ms"],
                                              "resources": r["resources"]}, indent=1, default=float))
    print(a.role, a.config, "processed", len(rows), "stage", {k: round(v, 2) for k, v in stage.items()}, flush=True)


def _first_frame(data: bytes):
    c, s = _open(data)
    f = next(iter(c.decode(s))).to_ndarray(format="rgb24")
    c.close()
    return f


def load_dets(role, config, stride):
    z = np.load(dets_path(role, config, stride))
    return {"frame": z["frame"], "box": z["box"], "score": z["score"]}, int(z["n_frames"])


# ------------------------------------------------------------------ development scoring / selection
DEV_ZONE_ROLES = ("dev_L10", "dev_L11")
DEV_ALL = ("dev_L09", "dev_L10", "dev_L11")


def cmd_devsel(a):
    from . import cag_eval as E
    from . import detectors as D
    out = {"protocol_sha256": sha256_file(PROTOCOL), "created_utc": utcnow(), "stride": a.stride, "candidates": {}}
    clips = {r: load_clip(r)[0] for r in DEV_ALL}
    out["geometry"] = {r: {"passes": c.geometry_ok, "consistency": c.consistency} for r, c in clips.items()}
    for cfg in a.configs:
        per = {}
        counts_by_tau = {float(t): np.zeros(4, np.int64) for t in E.TAUS}
        for r in DEV_ALL:
            dets, n = load_dets(r, cfg, a.stride)
            frames = np.arange(0, clips[r].n_frames, a.stride)
            frames = frames[frames < n]
            cc = E.zone_counts_only(clips[r], dets, E.TAUS, frames)
            for t, c in cc.items():
                counts_by_tau[t] += np.array([c["tp"], c["fn"], c["fp"], c["tn"]])
            per[r] = {"frames": frames, "dets": dets}
        rows = []
        for t, v in counts_by_tau.items():
            tpr = v[0] / max(v[0] + v[1], 1)
            fpr = v[2] / max(v[2] + v[3], 1)
            rows.append({"tau": t, "tpr": tpr, "fpr": fpr, "balanced_accuracy": (tpr + 1 - fpr) / 2})
        best = max(rows, key=lambda r: (r["balanced_accuracy"], -r["tau"]))
        tau = best["tau"]
        entry = {"best_tau": tau, "pooled_zone_dev": best, "tau_grid": rows, "clips": {}}
        for r in DEV_ALL:
            entry["clips"][r] = E.evaluate(clips[r], per[r]["dets"], tau, per[r]["frames"], boot=False)
        sj = json.loads((run_dir("dev_L11", cfg, f"offline_s{a.stride}") / "stage.json").read_text())
        entry["stage_ms_mean_dev_L11"] = sj["stage_ms_mean"]
        entry["pipeline_ms_mean_ex_decode_dev_L11"] = sj["pipeline_ms_mean_ex_decode"]
        out["candidates"][cfg] = entry
        print(cfg, "tau", tau, "pooled BA %.3f TPR %.3f FPR %.3f" % (best["balanced_accuracy"], best["tpr"], best["fpr"]), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dev_scores.json").write_text(json.dumps(out, indent=1, default=float))


# ------------------------------------------------------------------ real-time replay
def host_clock() -> dict:
    """Windows-host CPU clock/load (WSL2 cannot see real frequency). Sampled around each replay; thermal state is not observable."""
    import re
    import subprocess
    cmd = ("Get-CimInstance Win32_Processor | ForEach-Object { 'CUR=' + $_.CurrentClockSpeed + "
           "' MAX=' + $_.MaxClockSpeed + ' LOAD=' + $_.LoadPercentage }")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"CUR=(\d+) MAX=(\d+) LOAD=(\d+)", out)
        return {"cur_mhz": int(m.group(1)), "max_mhz": int(m.group(2)), "load_pct": int(m.group(3))} if m else {"raw": out.strip()}
    except Exception as e:
        return {"error": str(e)}


def replay_one(role, config, threads, rep, max_frames, tag):
    from . import detectors as D
    from analysis.edge_inference.run_stage_a import save_trace
    check_lock(role)
    video, _, _, _ = paths(role)
    data = video.read_bytes()
    engine, runner = D.build_engine(config, threads)
    clock0 = host_clock()
    man = manifest(role, config, "replay", threads, {"rep": rep, "tag": tag, "max_frames": max_frames, "queue_cap": 2,
                                                     "providers": runner.providers})
    r = run_replay(data, engine, queue_cap=2, speed=1.0, max_frames=max_frames)
    clock1 = host_clock()
    d = run_dir(role, config, f"replay_t{threads}_{tag}_r{rep}")
    d.mkdir(parents=True, exist_ok=True)
    summ = summarize_trace(r["trace"], 100.0, r["resources"]["wall_s"])
    summ["trace_violations"] = validate_trace(r["trace"])
    summ["resources"] = r["resources"]
    summ["host_clock_before"], summ["host_clock_after"] = clock0, clock1
    tr = r["trace"]
    proc = tr["dropped"] == 0
    summ["stage_ms_mean"] = {k: float(tr[k][proc].mean()) for k in ("decode_ms", "pre_ms", "infer_ms", "post_ms")}
    summ["dropped_fraction"] = float((tr["dropped"] == 1).mean())
    man["finished_utc"] = utcnow()
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    (d / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    save_trace(d / "trace.csv.gz", tr)
    p = cache_dir() / "cag" / "dets" / role / f"{config}_replay_t{threads}_{tag}_r{rep}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **r["dets"], n_frames=np.int32(r["n_frames"]))
    return summ


def cmd_replay(a):
    s = replay_one(a.role, a.config, a.threads, a.rep, a.max_frames, a.tag)
    keys = ("processed", "dropped", "dropped_fraction", "latency_ms", "stage_ms_mean", "trace_violations",
            "host_clock_before", "host_clock_after")
    print(json.dumps({k: s[k] for k in keys}, default=float))


def cmd_feasibility(a):
    """protocol_v2 real_time_feasibility_rule: 60 s paced replay of dev_L11 frames 0-1799; 4 threads, else 8."""
    out = {}
    for cfg in a.configs:
        out[cfg] = {}
        for th in (4, 8):
            s = replay_one("dev_L11", cfg, th, 0, 1800, "feas")
            ok = s["dropped_fraction"] <= 0.01 and s["latency_ms"]["p99"] <= 100.0
            out[cfg][str(th)] = {"dropped_fraction": s["dropped_fraction"], "latency_ms": s["latency_ms"], "passes": bool(ok),
                                 "stage_ms_mean": s["stage_ms_mean"]}
            print(cfg, th, "drop %.3f" % s["dropped_fraction"], "p99 %.1f" % s["latency_ms"]["p99"], "PASS" if ok else "fail", flush=True)
            if ok:
                break
        out[cfg]["deployment_threads"] = next((int(t) for t in ("4", "8") if t in out[cfg] and out[cfg][t]["passes"]), None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "feasibility.json").write_text(json.dumps({"protocol_sha256": sha256_file(PROTOCOL), "created_utc": utcnow(),
                                                          "host": host_state(), "results": out}, indent=1, default=float))


# ------------------------------------------------------------------ selection (this file's existence unlocks the holdout)
def cmd_select(a):
    dev = json.loads((RESULTS / "dev_scores.json").read_text())
    feas = json.loads((RESULTS / "feasibility.json").read_text())["results"]
    rows = {}
    for cfg, e in dev["candidates"].items():
        th = feas.get(cfg, {}).get("deployment_threads")
        rows[cfg] = {"pooled_ba": e["pooled_zone_dev"]["balanced_accuracy"], "tau": e["best_tau"],
                     "pipeline_ms_ex_decode": e["pipeline_ms_mean_ex_decode_dev_L11"], "deployment_threads": th,
                     "eligible": th is not None}
    elig = {k: v for k, v in rows.items() if v["eligible"]}
    best = max(v["pooled_ba"] for v in elig.values())
    tied = [k for k, v in elig.items() if v["pooled_ba"] >= best - 0.01]
    chosen = min(tied, key=lambda k: rows[k]["pipeline_ms_ex_decode"])
    sel = {"selected": chosen, "reference": "S0", "rule": "protocol_v2.json selection_rule", "table": rows, "tied_within_0.01": tied,
           "tau": {"S0": rows["S0"]["tau"], chosen: rows[chosen]["tau"]},
           "threads": {"S0": rows["S0"]["deployment_threads"], chosen: rows[chosen]["deployment_threads"]},
           "protocol_sha256": sha256_file(PROTOCOL), "dev_scores_sha256": sha256_file(RESULTS / "dev_scores.json"),
           "feasibility_sha256": sha256_file(RESULTS / "feasibility.json"), "created_utc": utcnow(),
           "holdout_clips": [r for r in CLIPS if r.startswith("hold_")]}
    (RESULTS / "dev_selection.json").write_text(json.dumps(sel, indent=1, default=float))
    print(json.dumps(sel, indent=1, default=float))


def cmd_holdout(a):
    """ONE scoring of S0 and the selected configuration per fresh holdout clip, at the frozen tau."""
    from . import cag_eval as E
    check_lock(a.role)
    sel = json.loads((RESULTS / "dev_selection.json").read_text())
    clip, _ = load_clip(a.role)
    res = {"role": a.role, "geometry_ok": clip.geometry_ok, "protocol_sha256": sha256_file(PROTOCOL),
           "dev_selection_sha256": sha256_file(RESULTS / "dev_selection.json"), "configs": {}}
    for cfg in dict.fromkeys(["S0", sel["selected"]]):
        dets, n = load_dets(a.role, cfg, 1)
        frames = np.arange(min(n, clip.n_frames))
        ev_ = E.evaluate(clip, dets, sel["tau"][cfg], frames, boot=True)
        z = ev_["zone"]
        ev_["quality_bar_met_on_this_layout"] = bool(z["tpr"] >= 0.90 and z["fpr"] <= 0.10 and
                                                     z["bootstrap"]["balanced_accuracy_ci"][0] >= 0.85) if z else None
        res["configs"][cfg] = ev_
        if z:
            print(a.role, cfg, "tau", sel["tau"][cfg], "BA %.3f TPR %.3f FPR %.3f" % (z["balanced_accuracy"], z["tpr"], z["fpr"]),
                  "bar", ev_["quality_bar_met_on_this_layout"], flush=True)
    (RESULTS / f"holdout_{a.role}.json").write_text(json.dumps(res, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect"); p.add_argument("config"); p.add_argument("role"); p.add_argument("--stride", type=int, default=3)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--max-frames", type=int, default=None)
    p = sub.add_parser("devsel"); p.add_argument("--stride", type=int, default=3)
    p.add_argument("--configs", nargs="+", default=["S0", "S1", "S2", "YT", "YS"])
    p = sub.add_parser("replay"); p.add_argument("config"); p.add_argument("role"); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--rep", type=int, default=0); p.add_argument("--max-frames", type=int, default=None); p.add_argument("--tag", default="t")
    p = sub.add_parser("feasibility"); p.add_argument("--configs", nargs="+", default=["S0", "S1", "S2", "YT", "YS"])
    sub.add_parser("select")
    p = sub.add_parser("holdout"); p.add_argument("role")
    a = ap.parse_args(argv)
    {"detect": cmd_detect, "devsel": cmd_devsel, "replay": cmd_replay, "feasibility": cmd_feasibility,
     "select": cmd_select, "holdout": cmd_holdout}[a.cmd](a)


if __name__ == "__main__":
    main()
