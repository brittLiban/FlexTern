"""
EXP-0004 driver: runs the configured phases and writes results/model_precision/.

Configurations (ternary transformer weights unchanged in all):
    baseline : BF16 head, DynamicCache (BF16 KV)
    A        : INT4 RTN head (g=128), DynamicCache
    B        : BF16 head, QuantizedCache int4 (quanto, g=64, residual 128)
    C        : INT4 RTN head, QuantizedCache int4

Phases (select with --phases): lmeval, incremental, generation, memory, latency, report.
Every run writes a manifest with the exact settings. Nothing here is a hardware result.

Usage (from repo root):
    python analysis/model_precision/run_experiment.py --phases lmeval --limit 20      # pipeline check
    python analysis/model_precision/run_experiment.py --phases lmeval                 # full tasks
    python analysis/model_precision/run_experiment.py --phases incremental --windows 8 --ctx 2048
    python analysis/model_precision/run_experiment.py --phases generation,memory,latency,report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS_DIR, SEED, cuda_mem, environment, load_model, read_json, reset_peak,  # noqa: E402
                    seed_everything, verify_checkpoint, write_json)
from head_quantization import quantize_lm_head, restore_lm_head  # noqa: E402

CONFIGS = {"baseline": ("bf16", "dynamic"),
           "A": ("int4", "dynamic"),
           "B": ("bf16", "int4"),          # transformers default grouping (axis 0, strided across the flattened cache)
           "B1": ("bf16", "int4_ax1"),     # per-token grouping (axis 1, 64 along head_dim)
           "C": ("int4", "int4_ax1"),      # head INT4 + per-token int4 KV
           "C0": ("int4", "int4")}         # head INT4 + default-axis int4 KV


def set_head(model, kind, state, group_size):
    """Switch the head between the original BF16 Linear and the INT4 RTN module, caching the original on CPU."""
    import torch
    if kind == "bf16":
        if state.get("head_is_int4"):
            restore_lm_head(model, state["original_head"], "cuda")
            state["head_is_int4"] = False
    else:
        if not state.get("head_is_int4"):
            if "original_head" not in state:
                original, qm, stats = quantize_lm_head(model, group_size)
                state["original_head"] = original
                state["head_stats"] = stats
                state["qm"] = qm
            else:
                model.lm_head = state["qm"]
            state["head_is_int4"] = True
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", default="lmeval,incremental,generation,memory,latency,report")
    ap.add_argument("--configs", default="baseline,A,B,B1,C")
    ap.add_argument("--limit", type=int, default=None, help="lm_eval per-task sample limit (PILOT if set)")
    ap.add_argument("--tasks", default="wikitext,hellaswag,piqa,arc_easy,arc_challenge,winogrande")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hellaswag-limit", type=int, default=2000, help="HellaSwag subset size (PILOT); 0 = full 10,042")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--contexts", default="128,512,2048,4096,8192,16384")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--cache-unpacked", action="store_true",
                    help="unpack ternary weights to BF16 once (numerically identical; ~2x faster decode; 3.2 GB resident)")
    args = ap.parse_args()
    phases = args.phases.split(",")
    configs = args.configs.split(",")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seed_everything(SEED)
    import torch
    device = "cuda"
    env = environment(); write_json("environment.json", env)
    ident = verify_checkpoint()
    model, tok = load_model(device)
    state = {}
    manifest = {"args": vars(args), "configs": CONFIGS, "checkpoint": ident, "env": env, "phases_run": []}
    if args.cache_unpacked:
        from common import cache_unpacked_ternary_weights
        manifest["ternary_unpack_cache"] = cache_unpacked_ternary_weights(model)
        manifest["ternary_unpack_cache"]["note"] = ("ternary weights unpacked once to BF16 for decode speed; outputs identical "
                                                   "(see unpack_cache_check.json); ternary bytes are reported from the packed size")
        print("ternary unpack cache:", manifest["ternary_unpack_cache"], flush=True)

    if "lmeval" in phases:
        from evaluate import run_lm_eval
        tasks = args.tasks.split(",")
        suffix = f"_limit{args.limit}" if args.limit else ""
        fname = f"lm_eval{suffix}.json"
        # Resumable: one (config, task) at a time, saved immediately, skipped if already present.
        try:
            lme = read_json(fname)
        except FileNotFoundError:
            lme = {}
        for cfg in configs:
            hk, kk = CONFIGS[cfg]
            # KV configs are identical to their head-equivalent for loglikelihood tasks; run B once on one task to demonstrate
            run_tasks = tasks if kk == "dynamic" else [tasks[0]]
            if kk != "dynamic" and cfg != "B":
                lme[cfg] = {"note": "identical to its head-equivalent by construction (no incremental cache in loglikelihood scoring); not re-run"}
                write_json(fname, lme)
                continue
            entry = lme.setdefault(cfg, {"head": hk, "kv_note": "loglikelihood scoring uses one forward pass; KV cache setting has no effect", "tasks": {}})
            for task in run_tasks:
                if task in entry["tasks"]:
                    print(f"== lm_eval {cfg}/{task}: already done, skipping ==", flush=True)
                    continue
                set_head(model, hk, state, args.group_size)
                limit = args.limit
                if limit is None and task == "hellaswag" and args.hellaswag_limit:
                    limit = args.hellaswag_limit
                print(f"== lm_eval {cfg}/{task} (head={hk}, limit={limit}) ==", flush=True)
                r = run_lm_eval(model, tok, [task], limit=limit, batch_size=args.batch_size)
                tr = r["tasks"][task]
                tr.update({"wall_s": r["wall_s"], "limit": limit, "pilot_subset": limit is not None, "max_length": r["max_length"],
                           "batch_size": r["batch_size"], "version": r["versions"].get(task)})
                entry["tasks"][task] = tr
                write_json(fname, lme)
                print(json.dumps(tr, indent=1), flush=True)
        manifest["phases_run"].append(f"lmeval{suffix}")

    if "incremental" in phases:
        from evaluate import wikitext_windows, single_forward_nll, incremental_nll
        windows, meta = wikitext_windows(tok, args.windows, args.ctx)
        fname = f"incremental_ctx{args.ctx}_w{args.windows}.json"
        try:
            inc = read_json(fname)
        except FileNotFoundError:
            inc = {"windows": meta, "prefill": args.prefill}
        if "single_forward_bf16_head" not in inc:
            set_head(model, "bf16", state, args.group_size)
            inc["single_forward_bf16_head"] = single_forward_nll(model, windows, args.prefill, device)
            write_json(fname, inc)
        print("single-forward reference ppl:", inc["single_forward_bf16_head"]["ppl"], flush=True)
        for cfg in configs:
            if cfg in inc:
                print(f"== incremental {cfg}: already done, skipping ==", flush=True)
                continue
            hk, kk = CONFIGS[cfg]
            set_head(model, hk, state, args.group_size)
            print(f"== incremental {cfg} (head={hk}, kv={kk}) ==", flush=True)
            r = incremental_nll(model, windows, args.prefill, kk, device,
                                progress=lambda i, n, v: print(f"  window {i}/{n} nll={v:.4f}", flush=True))
            r["head"] = hk
            inc[cfg] = r
            write_json(fname, inc)
            print(f"  ppl={r['ppl']:.4f} wall={r['wall_s']:.0f}s", flush=True)
        manifest["phases_run"].append(f"incremental_ctx{args.ctx}_w{args.windows}")

    if "generation" in phases:
        from evaluate import generate_sample, wikitext_windows
        prompts = {"short": "The three most important inventions of the twentieth century were",
                   "medium": None, "long": None}
        win2k, _ = wikitext_windows(tok, 1, 2048)
        prompts["medium"] = win2k[0, :1500]
        if model.config.max_position_embeddings >= 8192:
            win8k, _ = wikitext_windows(tok, 1, 6144)
            prompts["long"] = win8k[0]
        gen = {}
        for cfg in configs:
            hk, kk = CONFIGS[cfg]
            set_head(model, hk, state, args.group_size)
            gen[cfg] = {}
            for name, p in prompts.items():
                if p is None:
                    continue
                ids = tok(p, return_tensors="pt").input_ids[0] if isinstance(p, str) else p
                gen[cfg][name] = generate_sample(model, tok, ids, kk, args.new_tokens, device)
                print(f"[{cfg}/{name}] {gen[cfg][name]['decode_tokens_per_s']:.1f} tok/s :: {gen[cfg][name]['text'][:120]!r}", flush=True)
        write_json("generation.json", gen)
        manifest["phases_run"].append("generation")

    if "memory" in phases:
        from benchmark import kv_memory_sweep
        from evaluate import wikitext_windows
        contexts = [int(c) for c in args.contexts.split(",")]
        maxc = max(c for c in contexts if c <= model.config.max_position_embeddings)
        win, _ = wikitext_windows(tok, 1, maxc + 1)
        mem = {"contexts": contexts}
        for cfg in configs:
            hk, kk = CONFIGS[cfg]
            set_head(model, hk, state, args.group_size)
            reset_peak()
            mem[cfg] = {"head": hk, "kv": kk, "resident_after_config": cuda_mem(),
                        "head_bytes": state["head_stats"]["total_resident_bytes"] if hk == "int4" else model.lm_head.weight.numel() * 2,
                        "sweep": kv_memory_sweep(model, tok, win[0], kk, contexts, device)}
            print(f"[{cfg}] " + "; ".join(f"ctx{r['ctx']}: {r.get('cache_total_bytes', 0)/1e6:.1f}MB peak {r.get('peak_prefill_max_allocated_bytes', 0)/1e9:.2f}GB" for r in mem[cfg]["sweep"] if "ctx" in r), flush=True)
        write_json("memory.json", mem)
        manifest["phases_run"].append("memory")

    if "latency" in phases:
        from benchmark import latency
        from evaluate import wikitext_windows
        win, _ = wikitext_windows(tok, 1, 2048)
        try:
            lat = read_json("latency.json")
        except FileNotFoundError:
            lat = {}
        for cfg in configs:
            hk, kk = CONFIGS[cfg]
            set_head(model, hk, state, args.group_size)
            lat[cfg] = {"head": hk, "kv": kk}
            for plen in (128, 2048):
                lat[cfg][f"prompt{plen}"] = latency(model, tok, win[0, :plen], kk, args.new_tokens, device)
                print(f"[{cfg}] prompt {plen}: prefill {lat[cfg][f'prompt{plen}']['prefill_s_median']*1e3:.0f} ms, decode {lat[cfg][f'prompt{plen}']['decode_tokens_per_s_median']:.1f} tok/s", flush=True)
        write_json("latency.json", lat)
        manifest["phases_run"].append("latency")

    if "head_stats" in state:
        write_json("head_int4.json", {"method": "RTN asymmetric groupwise, fp16 scale, 4-bit zero (stored uint8), codes packed 2/byte along K; dequantize-to-BF16 per forward",
                                      **state["head_stats"]})
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    write_json("manifest_" + "_".join(phases) + ".json", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
