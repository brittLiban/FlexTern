"""Streaming replay harness for FT-0010 Stage A.

Replays an encoded video at its original frame timing: frame i is *released* at
t0 + pts_i (perf_counter clock). A feeder thread decodes at release time and pushes the
RGB frame into a bounded FIFO (capacity 2, drop-oldest when full); a worker thread runs
preprocess -> network -> postprocess. Every frame gets exactly one trace row, including
dropped ones. Latency = output_done - scheduled release, so decode, queueing and any
feeder slip all count.

The harness is runtime-agnostic: it takes an `Engine` with pre/infer/post callables, so
tests can drive it with fakes and B0/B1 differ only in the network inside the engine.
"""
from __future__ import annotations

import collections
import io
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

NS = 1_000_000_000
TRACE_FIELDS = ("frame", "pts", "sched_ns", "release_ns", "decode_ms", "start_ns", "pre_ms", "infer_ms",
                "post_ms", "done_ns", "dropped", "n_dets")


@dataclass
class Engine:
    name: str
    pre: Callable
    infer: Callable
    post: Callable  # (network_output, frame_w, frame_h) -> (boxes, scores)


def frame_index(pts: int, time_base: Fraction, fps: Fraction, first_pts: int = 0) -> int:
    """Timestamp-preserving frame index: round((pts - first) * time_base * fps)."""
    return int(round(float((pts - first_pts) * time_base * fps)))


def _open(video_bytes: bytes):
    import av
    c = av.open(io.BytesIO(video_bytes))
    s = c.streams.video[0]
    s.thread_type = "SLICE"
    s.codec_context.thread_count = 2
    return c, s


def count_packets(video_bytes: bytes) -> tuple[int, Fraction, Fraction, int, int, int]:
    c, s = _open(video_bytes)
    n, first = 0, None
    for p in c.demux(s):
        if p.pts is None:
            continue
        first = p.pts if first is None else min(first, p.pts)
        n += 1
    info = (n, Fraction(s.time_base), Fraction(s.average_rate), first or 0, s.codec_context.width, s.codec_context.height)
    c.close()
    return info


def _new_trace(n: int) -> dict:
    t = {k: np.zeros(n, dtype=np.float64 if k.endswith("_ms") else np.int64) for k in TRACE_FIELDS}
    t["frame"][:] = np.arange(n)
    t["dropped"][:] = -1  # -1 = never observed; must be 0/1 at the end
    return t


def _dets_to_arrays(dets: dict[int, tuple]) -> dict:
    fr, bx, sc = [], [], []
    for f in sorted(dets):
        b, s = dets[f]
        fr.append(np.full(len(s), f, np.int32))
        bx.append(b.reshape(-1, 4))
        sc.append(s)
    if not fr:
        return {"frame": np.zeros(0, np.int32), "box": np.zeros((0, 4), np.float32), "score": np.zeros(0, np.float32)}
    return {"frame": np.concatenate(fr), "box": np.concatenate(bx), "score": np.concatenate(sc)}


def warmup(engine: Engine, rgb: np.ndarray, n: int = 50):
    h, w = rgb.shape[:2]
    for _ in range(n):
        engine.post(engine.infer(engine.pre(rgb)), w, h)


class ResourceSampler:
    """Peak RSS (sampled) and process CPU time. Diagnostic only; not an energy measurement."""

    def __init__(self, period_s: float = 0.25):
        import psutil
        self.p = psutil.Process()
        self.period = period_s
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            self._stop.wait(self.period)

    def __enter__(self):
        ct = self.p.cpu_times()
        self.cpu0, self.wall0 = ct.user + ct.system, time.perf_counter()
        self.rss_start = self.p.memory_info().rss
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        ct = self.p.cpu_times()
        self.peak = max(self.peak, self.p.memory_info().rss)
        self.cpu_s = ct.user + ct.system - self.cpu0
        self.wall_s = time.perf_counter() - self.wall0

    def report(self) -> dict:
        return {"rss_start_mb": self.rss_start / 2**20, "rss_peak_mb": self.peak / 2**20,
                "process_cpu_s": self.cpu_s, "wall_s": self.wall_s,
                "avg_busy_cores": self.cpu_s / self.wall_s if self.wall_s else None}


def run_offline(video_bytes: bytes, engine: Engine, max_frames: int | None = None):
    """Every frame, no pacing, same decode/pre/infer/post path. Used for quality + parity + determinism."""
    n, tb, fps, first, W, H = count_packets(video_bytes)
    if max_frames:
        n = min(n, max_frames)
    trace = _new_trace(n)
    dets: dict[int, tuple] = {}
    c, s = _open(video_bytes)
    pc = time.perf_counter_ns
    i = 0
    with ResourceSampler() as rs:
        for pkt in c.demux(s):
            if pkt.pts is None:
                continue
            t0 = pc()
            frames = pkt.decode()
            for fr in frames:
                if i >= n:
                    break
                rgb = fr.to_ndarray(format="rgb24")
                t1 = pc()
                x = engine.pre(rgb)
                t2 = pc()
                out = engine.infer(x)
                t3 = pc()
                b, sc = engine.post(out, W, H)
                t4 = pc()
                idx = frame_index(fr.pts, tb, fps, first)
                trace["pts"][i] = fr.pts
                trace["frame"][i] = idx
                trace["decode_ms"][i] = (t1 - t0) / 1e6
                trace["pre_ms"][i] = (t2 - t1) / 1e6
                trace["infer_ms"][i] = (t3 - t2) / 1e6
                trace["post_ms"][i] = (t4 - t3) / 1e6
                trace["dropped"][i] = 0
                trace["n_dets"][i] = len(sc)
                dets[idx] = (b, sc)
                i += 1
            if i >= n:
                break
    c.close()
    return {"trace": {k: v[:i] for k, v in trace.items()}, "dets": _dets_to_arrays(dets),
            "resources": rs.report(), "frame_w": W, "frame_h": H, "n_frames": i}


def run_replay(video_bytes: bytes, engine: Engine, queue_cap: int = 2, speed: float = 1.0,
               max_frames: int | None = None, do_warmup: int = 50):
    n, tb, fps, first, W, H = count_packets(video_bytes)
    if max_frames:
        n = min(n, max_frames)
    pc = time.perf_counter_ns
    if do_warmup:
        c0, s0 = _open(video_bytes)
        f0 = next(iter(c0.decode(s0))).to_ndarray(format="rgb24")
        c0.close()
        warmup(engine, f0, do_warmup)
    trace = _new_trace(n)
    dets: dict[int, tuple] = {}
    q: collections.deque = collections.deque()
    cond = threading.Condition()
    state = {"done": False}
    t0_ns = pc() + 300_000_000  # first release 300 ms after setup so early frames are not late by construction

    def feeder():
        c, s = _open(video_bytes)
        sent = 0
        try:
            for pkt in c.demux(s):
                if pkt.pts is None:
                    continue
                if sent >= n:
                    break
                idx = frame_index(pkt.pts, tb, fps, first)
                sched = int(float((pkt.pts - first) * tb) * NS / speed)
                target = t0_ns + sched
                while True:  # sleep coarsely, spin the last 0.5 ms
                    rem = target - pc()
                    if rem <= 0:
                        break
                    if rem > 1_500_000:
                        time.sleep((rem - 1_000_000) / 1e9)
                rel = pc()
                d0 = pc()
                frames = pkt.decode()
                for fr in frames:
                    rgb = fr.to_ndarray(format="rgb24")
                    d1 = pc()
                    fi = frame_index(fr.pts, tb, fps, first)
                    trace["pts"][fi] = fr.pts
                    trace["sched_ns"][fi] = target - t0_ns
                    trace["release_ns"][fi] = rel - t0_ns
                    trace["decode_ms"][fi] = (d1 - d0) / 1e6
                    with cond:
                        if len(q) >= queue_cap:
                            old = q.popleft()
                            trace["dropped"][old[0]] = 1
                            trace["done_ns"][old[0]] = pc() - t0_ns  # time at which it was discarded
                        q.append((fi, rgb))
                        cond.notify()
                    sent += 1
                    break  # no B-frames in these streams: one frame per packet (asserted via trace row count)
        finally:
            with cond:
                state["done"] = True
                cond.notify_all()
            c.close()

    def worker():
        while True:
            with cond:
                while not q and not state["done"]:
                    cond.wait()
                if not q and state["done"]:
                    return
                fi, rgb = q.popleft()
            s0 = pc()
            x = engine.pre(rgb)
            s1 = pc()
            out = engine.infer(x)
            s2 = pc()
            b, sc = engine.post(out, W, H)
            s3 = pc()
            trace["start_ns"][fi] = s0 - t0_ns
            trace["pre_ms"][fi] = (s1 - s0) / 1e6
            trace["infer_ms"][fi] = (s2 - s1) / 1e6
            trace["post_ms"][fi] = (s3 - s2) / 1e6
            trace["done_ns"][fi] = s3 - t0_ns
            trace["dropped"][fi] = 0
            trace["n_dets"][fi] = len(sc)
            dets[fi] = (b, sc)

    with ResourceSampler() as rs:
        tf, tw = threading.Thread(target=feeder), threading.Thread(target=worker)
        tw.start()
        tf.start()
        tf.join()
        tw.join()
    return {"trace": trace, "dets": _dets_to_arrays(dets), "resources": rs.report(), "frame_w": W, "frame_h": H,
            "n_frames": n, "t0_ns": t0_ns, "speed": speed, "queue_cap": queue_cap}


def validate_trace(trace: dict) -> list[str]:
    """Invariants every replay trace must satisfy; returns a list of violations (empty = valid)."""
    bad = []
    n = len(trace["frame"])
    if not np.array_equal(trace["frame"], np.arange(n)):
        bad.append("frame indices are not 0..n-1 in order")
    if (trace["dropped"] < 0).any():
        bad.append("frames never observed (dropped == -1)")
    proc = trace["dropped"] == 0
    if (trace["done_ns"][proc] < trace["start_ns"][proc]).any():
        bad.append("done before start")
    if (trace["start_ns"][proc] < trace["release_ns"][proc]).any():
        bad.append("processing started before frame release")
    if (np.diff(trace["sched_ns"]) <= 0).any():
        bad.append("scheduled release times not strictly increasing")
    return bad


def summarize_trace(trace: dict, deadline_ms: float, wall_s: float | None = None) -> dict:
    n = len(trace["frame"])
    proc = trace["dropped"] == 0
    lat = (trace["done_ns"][proc] - trace["sched_ns"][proc]) / 1e6
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")  # noqa: E731
    on_time = int((lat <= deadline_ms).sum())
    slip = (trace["release_ns"] - trace["sched_ns"]) / 1e6
    stages = {}
    for k in ("decode_ms", "pre_ms", "infer_ms", "post_ms"):
        a = trace[k][proc] if k != "decode_ms" else trace[k]
        stages[k] = {"mean": float(np.mean(a)), "p50": q(a, 50), "p95": q(a, 95), "p99": q(a, 99)}
    qwait = (trace["start_ns"][proc] - trace["release_ns"][proc]) / 1e6
    return {
        "n_frames": n, "processed": int(proc.sum()), "dropped": int((trace["dropped"] == 1).sum()),
        "drop_fraction": float((trace["dropped"] == 1).mean()),
        "latency_ms": {"mean": float(lat.mean()) if len(lat) else float("nan"), "p50": q(lat, 50), "p95": q(lat, 95),
                       "p99": q(lat, 99), "max": float(lat.max()) if len(lat) else float("nan")},
        "deadline_ms": deadline_ms, "deadline_missed_processed": int((lat > deadline_ms).sum()),
        "on_time_fraction_of_all_frames": on_time / n if n else float("nan"),
        "queue_wait_ms_p95": q(qwait, 95), "feeder_slip_ms_p99": q(slip, 99), "feeder_slip_ms_max": float(slip.max()),
        "stage_ms": stages, "processed_fps": (int(proc.sum()) / wall_s) if wall_s else None,
    }
