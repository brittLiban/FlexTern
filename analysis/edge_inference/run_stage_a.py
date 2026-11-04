"""FT-0010 Stage A orchestrator.

Subcommands (each is one fresh process so runs do not contaminate each other):
  prepare                      export ONNX, anchors, model/clip identity -> results/edge_inference/identity.json
  parity                       torch-vs-ORT head tensors; official model(img) vs shared pipeline (dev clip)
  offline  ROLE RUNTIME [--rep N]   every frame, no pacing: quality, determinism
  calibrate                    pick the operating threshold on the calibration clip (B0 offline run required)
  replay   ROLE RUNTIME [--rep N]   real-time paced replay: latency, drops, memory, stage timing
  align                        ground-truth/video frame-alignment sweep on the development clip

ROLE in {calibration, development, holdout}; RUNTIME in {B0, B1}.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import evaluate as ev
from . import gt as gtmod
from . import pipeline
from .common import PKG, PROTOCOL_PATH, RESULTS, cache_dir, environment, load_protocol, sha256_bytes, sha256_file
from .replay import Engine, run_offline, run_replay, summarize_trace, validate_trace

RUNS = RESULTS / "runs"
RAW = cache_dir() / "runs"  # full-resolution detections (not committed); hashes are in the manifests


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def clip_cfg(role: str) -> dict:
    return load_protocol()["splits"][role]


def clip_paths(role: str) -> tuple[Path, Path]:
    c = clip_cfg(role)
    video = cache_dir() / "raw" / c["split_dir"] / c["scene"] / "videos" / f"{c['camera']}.mp4"
    gtp = cache_dir() / "gt" / c["scene"] / f"{c['camera']}.npz"
    return video, gtp


def clip_name(role: str) -> str:
    c = clip_cfg(role)
    return f"{role}_{c['scene']}_{c['camera']}"


BATTERY_STATUS = {1: "discharging (on battery)", 2: "AC power", 3: "fully charged (AC)", 4: "low", 5: "critical",
                  6: "charging (AC)", 7: "charging, high", 8: "charging, low", 9: "charging, critical", 11: "partially charged (AC)"}


def host_state() -> dict:
    """Power plan and AC/battery state at run start (raw Windows codes are kept; the label is derived)."""
    import re
    d = {"power_plan": "unknown", "battery_status_code": None, "battery_status": "unknown", "battery_charge_pct": None}
    try:
        cmd = ('Write-Output ("PLAN=" + (powercfg /getactivescheme)); '
               '$b = Get-CimInstance Win32_Battery; Write-Output ("BATT=" + $b.BatteryStatus + " PCT=" + $b.EstimatedChargeRemaining)')
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"PLAN=.*?\((.*?)\)", out)
        d["power_plan"] = m.group(1) if m else "unknown"
        m = re.search(r"BATT=(\d+) PCT=(\d+)", out)
        if m:
            d["battery_status_code"], d["battery_charge_pct"] = int(m.group(1)), int(m.group(2))
            d["battery_status"] = BATTERY_STATUS.get(d["battery_status_code"], "other")
    except Exception:
        pass
    d["note"] = "WSL2 guest under Hyper-V on a laptop; CPU governor, frequency, thermal state and Windows process priority are NOT controlled"
    return d


def build_engine(runtime: str, threads: int):
    from . import models
    anchors = np.load(models.ANCHORS_PATH)
    if runtime == "B0":
        model, _ = models.load_official(threads)
        runner = models.TorchRunner(model, threads)
    elif runtime == "B1":
        runner = models.OrtRunner(models.onnx_cache_path(), threads)  # torch is never imported on this path
    else:
        raise ValueError(runtime)
    return Engine(runtime, pipeline.preprocess, runner, lambda out, w, h: pipeline.postprocess(out[0], out[1], anchors, w, h)), runner


def manifest(role: str, runtime: str, mode: str, rep: int, extra: dict | None = None) -> dict:
    from . import models
    proto = load_protocol()
    video, gtp = clip_paths(role)
    gt = gtmod.load_camera_gt(gtp)
    c = clip_cfg(role)
    raw_gt = cache_dir() / "raw" / c["split_dir"] / c["scene"] / "ground_truth.json"
    m = {
        "started_utc": utcnow(), "role": role, "clip": clip_name(role), "runtime": runtime, "mode": mode, "repeat": rep,
        "command": " ".join(["python", "-m", "analysis.edge_inference.run_stage_a"] + sys.argv[1:]),
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "protocol_id": proto["protocol_id"],
        "video_sha256": sha256_file(video), "video_bytes": video.stat().st_size,
        "gt_json_sha256": sha256_file(raw_gt) if raw_gt.exists() else None,
        "gt_camera_content_sha256": gtmod.content_sha256(gt),
        "model": models.model_identity(),
        "threads": proto["runtimes"]["threads"], "seeds": proto["seeds"],
        "environment": environment(), "host": host_state(),
    }
    if extra:
        m.update(extra)
    return m


def _json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def save_trace(path: Path, trace: dict):
    keys = list(trace)
    with gzip.open(path, "wt") as f:
        f.write(",".join(keys) + "\n")
        for i in range(len(trace["frame"])):
            f.write(",".join(repr(float(trace[k][i])) if k.endswith("_ms") else str(int(trace[k][i])) for k in keys) + "\n")


def save_dets(path: Path, dets: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame=dets["frame"], box=dets["box"], score=dets["score"])


def dets_sha(dets: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in ("frame", "box", "score"):
        h.update(np.ascontiguousarray(dets[k]).tobytes())
    return h.hexdigest()


def tau_value() -> float | None:
    p = RESULTS / "calibration.json"
    return json.loads(p.read_text())["tau"] if p.exists() else None


def score_run(role: str, dets: dict, tau: float | None, n_frames: int, frames=None) -> tuple[ev.ClipScore, dict]:
    proto = load_protocol()
    _, gtp = clip_paths(role)
    gt = gtmod.load_camera_gt(gtp)
    sc = proto["scoring"]
    cs = ev.score_clip(n_frames, dets, gt, iou_thr=sc["iou_threshold"], frames=frames)
    strata = {k: tuple(v) for k, v in sc["size_strata_native_height_px"].items()}
    res = ev.clip_metrics(cs, tau if tau is not None else 0.5, strata)
    res["ap50_all_sizes_policy"] = "size floor applied (GT<32 px height ignored)"
    # also the all-size variant (no ignore floor) as a transparent sensitivity number
    cs_all = ev.score_clip(n_frames, dets, gt, iou_thr=sc["iou_threshold"], small_h=0.0, frames=frames)
    res["ap50_no_size_floor"] = ev.ap_from_arrays(cs_all.det_score, cs_all.det_flag, None, int((~cs_all.gt_ignore).sum()))
    res["operating_point_no_size_floor"] = ev.operating_point(cs_all, tau if tau is not None else 0.5)
    return cs, res


# ------------------------------------------------------------------ commands

def cmd_prepare(a):
    from . import models
    model, _ = models.load_official(4)
    models.get_anchors(model)
    p = models.onnx_cache_path()
    if not p.exists():
        models.export_onnx(model, p)
    ident = models.model_identity()
    ident["anchors_shape"] = list(np.load(models.ANCHORS_PATH).shape)
    clips = {}
    for role in ("calibration", "development", "holdout"):
        video, gtp = clip_paths(role)
        gt = gtmod.load_camera_gt(gtp)
        c = clip_cfg(role)
        clips[role] = {**c, "video_sha256": sha256_file(video), "video_bytes": video.stat().st_size,
                       "gt_camera_content_sha256": gtmod.content_sha256(gt), "gt_n_frames": gt["n_frames"],
                       "gt_person_size_stats": gtmod.size_stats(gt), "gt_class_counts": {k: int((gt["cls"] == v).sum()) for k, v in gtmod.CLASS_IDS.items() if (gt["cls"] == v).any()}}
    _json(RESULTS / "identity.json", {"model": ident, "clips": clips, "environment": environment()})
    print(json.dumps(ident, indent=1))


def cmd_parity(a):
    """(1) raw head tensors torch vs ORT; (2) official model(img) vs shared-pipeline B0 and B1."""
    import torch
    from . import models
    import av
    threads = load_protocol()["runtimes"]["threads"]["intra_op"]
    role = "development"
    video, gtp = clip_paths(role)
    model, _ = models.load_official(threads)
    anchors = np.load(models.ANCHORS_PATH)
    tr, ort_r = models.TorchRunner(model, threads), models.OrtRunner(models.onnx_cache_path(), threads)
    stride = load_protocol()["parity_tolerances"]["official_vs_shared_frame_stride"]
    c = av.open(str(video))
    s = c.streams.video[0]
    max_cls, max_box, n_cmp, frames = 0.0, 0.0, 0, []
    rows, off_dets, b0_dets, b1_dets = [], {"frame": [], "box": [], "score": []}, {"frame": [], "box": [], "score": []}, {"frame": [], "box": [], "score": []}
    n_total = 0
    t_start = time.time()
    for i, fr in enumerate(c.decode(s)):
        n_total += 1
        if i % stride or i >= 2000:
            continue
        rgb = fr.to_ndarray(format="rgb24")
        h, w = rgb.shape[:2]
        x = pipeline.preprocess(rgb)
        c0, b0 = tr(x)
        c1, b1 = ort_r(x)
        max_cls = max(max_cls, float(np.abs(c0 - c1).max()))
        max_box = max(max_box, float(np.abs(b0 - b1).max()))
        for name, (cl, bb) in (("b0", (c0, b0)), ("b1", (c1, b1))):
            bx, sc = pipeline.postprocess(cl, bb, anchors, w, h)
            d = b0_dets if name == "b0" else b1_dets
            d["frame"].append(np.full(len(sc), i, np.int32)); d["box"].append(bx); d["score"].append(sc)
        with torch.inference_mode():
            o = model([torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0])[0]
        m = o["labels"] == 1
        off_dets["frame"].append(np.full(int(m.sum()), i, np.int32)); off_dets["box"].append(o["boxes"][m].numpy()); off_dets["score"].append(o["scores"][m].numpy())
        frames.append(i)
    c.close()
    cat = lambda d: {"frame": np.concatenate(d["frame"]), "box": np.concatenate(d["box"]).reshape(-1, 4), "score": np.concatenate(d["score"])}  # noqa: E731
    off, b0d, b1d = cat(off_dets), cat(b0_dets), cat(b1_dets)
    frames = np.asarray(frames)
    n_frames = int(frames.max()) + 1
    res = {"frames_compared": len(frames), "stride": stride,
           "head_max_abs_diff_cls_logits_torch_vs_ort": max_cls, "head_max_abs_diff_bbox_regression_torch_vs_ort": max_box}
    ap = {}
    for name, d in (("official", off), ("shared_B0", b0d), ("shared_B1", b1d)):
        cs, r = score_run(role, d, 0.5, n_frames, frames)
        ap[name] = {"ap50": r["ap50"], "n_dets": int(len(d["score"]))}
    res["ap50"] = ap
    # detection-level comparison at score >= 0.3 between official and shared B0 / shared B0 and B1
    def compare(d1, d2, thr):
        same_count, maxdev, n = 0, 0.0, 0
        for f in frames:
            m1, m2 = (d1["frame"] == f) & (d1["score"] >= thr), (d2["frame"] == f) & (d2["score"] >= thr)
            b1_, b2_ = d1["box"][m1], d2["box"][m2]
            n += 1
            if len(b1_) == len(b2_):
                same_count += 1
                if len(b1_):
                    iou = ev.box_iou(b1_, b2_)
                    j = iou.argmax(axis=1)
                    maxdev = max(maxdev, float(np.abs(b1_ - b2_[j]).max()))
        return {"frames": n, "frames_same_detection_count": same_count, "max_box_deviation_px_native": maxdev}
    res["compare_official_vs_shared_B0_score_ge_0.3"] = compare(off, b0d, 0.3)
    res["compare_shared_B0_vs_B1_score_ge_0.3"] = compare(b0d, b1d, 0.3)
    tol = load_protocol()["parity_tolerances"]
    res["tolerances"] = tol
    res["pass"] = {
        "head_tensors": max(max_cls, max_box) <= tol["raw_head_max_abs_diff_torch_vs_ort"],
        "ap50_torch_vs_ort": abs(ap["shared_B0"]["ap50"] - ap["shared_B1"]["ap50"]) <= tol["ap50_abs_diff_torch_vs_ort"],
        "ap50_official_vs_shared": abs(ap["official"]["ap50"] - ap["shared_B0"]["ap50"]) <= tol["ap50_abs_diff_official_vs_shared_pipeline"],
    }
    res["elapsed_s"] = time.time() - t_start
    res["environment"] = environment()
    _json(RESULTS / "parity.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "environment"}, indent=1, default=float))


def run_dir(role, runtime, mode, rep) -> Path:
    return RUNS / clip_name(role) / runtime / f"{mode}_r{rep}"


def cmd_offline(a):
    proto = load_protocol()
    if a.role == "holdout" and tau_value() is None:
        sys.exit("holdout is locked until the operating threshold is fixed on the calibration clip (run `calibrate`)")
    th = proto["runtimes"]["threads"]["intra_op"]
    video, _ = clip_paths(a.role)
    data = video.read_bytes()
    engine, runner = build_engine(a.runtime, th)
    man = manifest(a.role, a.runtime, "offline", a.rep, {"providers": getattr(runner, "providers", None), "max_frames": a.max_frames})
    r = run_offline(data, engine, a.max_frames)
    d = run_dir(a.role, a.runtime, "offline", a.rep)
    d.mkdir(parents=True, exist_ok=True)
    tau = tau_value()
    cs, res = score_run(a.role, r["dets"], tau, r["n_frames"])
    res["tau_used"] = tau
    res["bootstrap"] = ev.block_bootstrap(cs, tau if tau is not None else 0.5, n_boot=1000, seed=proto["seeds"]["bootstrap"]) if tau is not None else None
    res["detections_sha256"] = dets_sha(r["dets"])
    res["n_detections"] = int(len(r["dets"]["score"]))
    res["resources"] = r["resources"]
    res["stage_ms"] = {k: {"mean": float(np.mean(r["trace"][k])), "p50": float(np.percentile(r["trace"][k], 50)), "p95": float(np.percentile(r["trace"][k], 95))} for k in ("decode_ms", "pre_ms", "infer_ms", "post_ms")}
    res["offline_throughput_fps"] = r["n_frames"] / r["resources"]["wall_s"]
    man["finished_utc"] = utcnow()
    _json(d / "manifest.json", man)
    _json(d / "summary.json", res)
    save_trace(d / "trace.csv.gz", r["trace"])
    save_dets(RAW / clip_name(a.role) / a.runtime / f"offline_r{a.rep}_dets.npz", r["dets"])
    print(json.dumps({k: res[k] for k in ("ap50", "ap50_no_size_floor", "precision", "recall", "f1", "tau_used", "offline_throughput_fps", "detections_sha256")}, indent=1, default=float))


def cmd_calibrate(a):
    proto = load_protocol()
    dets = np.load(RAW / clip_name("calibration") / "B0" / "offline_r0_dets.npz")
    dets = {k: dets[k] for k in dets.files}
    n_frames = int(gtmod.load_camera_gt(clip_paths("calibration")[1])["n_frames"])
    cs, _ = score_run("calibration", dets, 0.5, n_frames)
    tau, rows = ev.best_f1_threshold(cs)
    out = {"tau": tau, "selected_on": clip_name("calibration"), "runtime": "B0 offline", "rule": proto["scoring"]["operating_threshold"]["rule"],
           "protocol_sha256": sha256_file(PROTOCOL_PATH), "grid": rows, "created_utc": utcnow(), "detections_sha256": dets_sha(dets)}
    _json(RESULTS / "calibration.json", out)
    print("tau =", tau)
    for r in rows:
        print(f"  tau={r['tau']:.2f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")


def cmd_replay(a):
    proto = load_protocol()
    th = proto["runtimes"]["threads"]["intra_op"]
    st = proto["streaming"]
    tau = tau_value()
    if tau is None:
        sys.exit("run `calibrate` first: the operating threshold must be fixed before any replay is scored")
    video, _ = clip_paths(a.role)
    data = video.read_bytes()
    engine, runner = build_engine(a.runtime, th)
    man = manifest(a.role, a.runtime, "replay", a.rep, {"providers": getattr(runner, "providers", None), "speed": a.speed,
                                                         "queue_cap": 2, "max_frames": a.max_frames})
    r = run_replay(data, engine, queue_cap=2, speed=a.speed, max_frames=a.max_frames)
    d = run_dir(a.role, a.runtime, "replay" if a.speed == 1.0 else f"replay_x{a.speed:g}", a.rep)
    d.mkdir(parents=True, exist_ok=True)
    bad = validate_trace(r["trace"])
    summ = summarize_trace(r["trace"], st["deadline_ms"], r["resources"]["wall_s"])
    cs, q = score_run(a.role, r["dets"], tau, r["n_frames"])
    summ["quality_including_drops"] = {k: q[k] for k in ("ap50", "ap50_no_size_floor", "precision", "recall", "f1", "tp", "fp", "fn", "tau")}
    summ["trace_violations"] = bad
    summ["resources"] = r["resources"]
    summ["detections_sha256"] = dets_sha(r["dets"])
    summ["speed"] = a.speed
    # quality of the frames that were actually processed (drops excluded from the denominator)
    proc = np.nonzero(r["trace"]["dropped"] == 0)[0]
    if len(proc) < r["n_frames"]:
        _, qp = score_run(a.role, r["dets"], tau, r["n_frames"], frames=proc)
        summ["quality_processed_frames_only"] = {k: qp[k] for k in ("ap50", "precision", "recall", "f1")}
    man["finished_utc"] = utcnow()
    _json(d / "manifest.json", man)
    _json(d / "summary.json", summ)
    save_trace(d / "trace.csv.gz", r["trace"])
    save_dets(RAW / clip_name(a.role) / a.runtime / f"{d.name}_dets.npz", r["dets"])
    print(json.dumps({k: summ[k] for k in ("processed", "dropped", "latency_ms", "deadline_missed_processed", "feeder_slip_ms_p99", "trace_violations", "quality_including_drops")}, indent=1, default=float))
    print("resources:", json.dumps(summ["resources"], default=float))


def cmd_align(a):
    """Score the B0 offline dev detections against GT shifted by -3..+3 frames; the peak must be at 0."""
    proto = load_protocol()
    dets = np.load(RAW / clip_name("development") / "B0" / "offline_r0_dets.npz")
    dets = {k: dets[k] for k in dets.files}
    gt = gtmod.load_camera_gt(clip_paths("development")[1])
    n = gt["n_frames"] - 4
    rows = {}
    for sh in range(-3, 4):
        cs = ev.score_clip(n, dets, gt, frame_shift=sh, frames=np.arange(3, n))
        rows[sh] = ev.ap_from_arrays(cs.det_score, cs.det_flag, None, int((~cs.gt_ignore).sum()))
    best = max(rows, key=rows.get)
    out = {"clip": clip_name("development"), "ap50_by_gt_frame_shift": rows, "argmax_shift": best, "aligned": best == 0}
    _json(RESULTS / "alignment_sweep.json", out)
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    sp.add_parser("parity")
    sp.add_parser("calibrate")
    sp.add_parser("align")
    for n in ("offline", "replay"):
        p = sp.add_parser(n)
        p.add_argument("role", choices=["calibration", "development", "holdout"])
        p.add_argument("runtime", choices=["B0", "B1"])
        p.add_argument("--rep", type=int, default=0)
        p.add_argument("--max-frames", type=int, default=None)
        if n == "replay":
            p.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "parity": cmd_parity, "offline": cmd_offline, "calibrate": cmd_calibrate,
     "replay": cmd_replay, "align": cmd_align}[a.cmd](a)


if __name__ == "__main__":
    main()
