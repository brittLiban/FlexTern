"""MEASURED batch-1 CPU per-control-step latency of the PyTorch reference model (EXP-0012, rebuilt on the shared runner in EXP-0013).

Quantized arms have no packed kernel in this code base: their fake-quant graph is slower than FP32 and is NOT timed
(it would measure simulation overhead, not an implementation). Only the plain FP checkpoint's deployment path is timed:
pooled-pixel features (trigger only) + (optional) encoder + GRU cell + policy head (+ trigger head). The predictor head and EMA
encoder are training/analysis-only and excluded. Run on AC power with nothing else on the CPU (KI-021).

EXP-0013 Phase 1: the step is `infer.RecurrentRunner` -- the same code `evaluate.rollout` uses -- so trigger decisions and
recurrent updates are identical in both paths, `fixed:K` is supported, blackout frames are never encoded, and `full` no longer
pays for the 8x8 pooling it does not use. Latency is split into trigger / encoder / update time (perf_counter, batch 1).
Usage: python -m analysis.predictive_state.latency CKPT.pt OUT.json
"""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import torch

from . import env as E
from . import infer as I
from . import train as T


def pct(x):
    x = np.asarray(x) * 1e3
    return {"p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)), "max_ms": float(x.max()), "mean_ms": float(x.mean()), "n": len(x)}


@torch.no_grad()
def run(net, trigger, threads, episodes=15, seed=31, cond="DIST"):
    torch.set_num_threads(threads)
    lat, enc_flags = [], []
    comp = {"trigger": 0.0, "encoder": 0.0, "update": 0.0}
    tot = I.Counters()
    for ep in range(episodes):
        env = E.BatchEnv(1, "test", E.TEST_CONDS[cond], seed=seed + ep)
        obs = env.reset()
        runner = I.RecurrentRunner(net, trigger, 1, "cpu", timing=True)
        for t in range(E.T_PICK):
            if env.done[0]:
                break
            img = torch.from_numpy(obs["img"]); prop = torch.from_numpy(obs["prop"]); task = torch.from_numpy(obs["task"])
            valid = torch.from_numpy(obs["valid"]); alive = torch.ones(1, dtype=torch.bool)
            t0 = time.perf_counter()
            act, enc_mask = runner.step(img, valid, prop, task, alive)
            lat.append(time.perf_counter() - t0)
            enc_flags.append(bool(enc_mask[0]))
            obs, _ = env.step(act.numpy())
        c = runner.counters
        comp["trigger"] += c.t_trigger; comp["encoder"] += c.t_encoder; comp["update"] += c.t_update
        for f in ("steps", "rec_updates", "active_steps", "enc_calls", "enc_samples", "blackout_rows", "trigger_skipped_rows"):
            setattr(tot, f, getattr(tot, f) + getattr(c, f))
    out = pct(lat)
    out["enc_frac"] = float(np.mean(enc_flags))
    out["enc_step"] = pct([l for l, f in zip(lat, enc_flags) if f]) if any(enc_flags) else None
    out["skip_step"] = pct([l for l, f in zip(lat, enc_flags) if not f]) if not all(enc_flags) else None
    out["component_mean_ms"] = {k: 1e3 * v / max(len(lat), 1) for k, v in comp.items()}
    out["counters"] = tot.as_dict()
    return out


def main():
    ck, out = sys.argv[1], sys.argv[2]
    taus = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    net, _ = T.load_net(ck, "cpu")
    res = {"checkpoint": ck, "platform": platform.platform(), "torch": torch.__version__, "cpu_count": torch.get_num_threads(), "runs": {}}
    for th in (1, 4):
        for trig in ["full"] + list(taus.keys()):
            spec = "full" if trig == "full" else taus[trig]
            res["runs"][f"threads{th}|{trig}"] = run(net, spec, th)
            print(th, trig, res["runs"][f"threads{th}|{trig}"]["p50_ms"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
