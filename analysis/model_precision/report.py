"""
EXP-0004 report generator: joins results/model_precision/*.json into
quality_comparison.csv, memory_comparison.csv, latency_comparison.csv and candidate_format.md.

Usage (from repo root): python analysis/model_precision/report.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, read_json  # noqa: E402

CONFIG_DESC = {"baseline": "BF16 head, BF16 KV", "A": "INT4 head (RTN g128), BF16 KV",
               "B": "BF16 head, int4 KV (hqq g64 axis0 strided, 128-token BF16 residual)",
               "B1": "BF16 head, int4 KV (hqq g64 axis1 per-token, 128-token BF16 residual)",
               "C": "INT4 head + int4 KV (axis1)", "C0": "INT4 head + int4 KV (axis0)"}


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(RESULTS_DIR, "*.json"))}
    quality = []
    # --- lm_eval (head configurations)
    lme_files = sorted(f for f in files if f.startswith("lm_eval"))
    for f in lme_files:
        d = json.load(open(files[f], encoding="utf-8"))
        pilot = "limit" in f
        base = d.get("baseline", {}).get("tasks", {})
        for cfg, r in d.items():
            if "tasks" not in r:
                continue
            for task, m in r["tasks"].items():
                for metric, val in m.items():
                    if metric.endswith("stderr") or metric == "n_samples" or not isinstance(val, (int, float)):
                        continue
                    b = base.get(task, {}).get(metric)
                    se = m.get(f"{metric}_stderr") if isinstance(m.get(f"{metric}_stderr"), (int, float)) else None
                    quality.append({"source": f, "pilot": pilot, "config": cfg, "description": CONFIG_DESC.get(cfg, ""), "task": task, "metric": metric,
                                    "n_samples": m.get("n_samples"), "baseline": b, "value": val,
                                    "abs_diff": (val - b) if b is not None else None,
                                    "rel_diff_pct": (100 * (val - b) / b) if b not in (None, 0) else None,
                                    "pp_diff": (100 * (val - b)) if (b is not None and metric.startswith("acc")) else None,
                                    "stderr": se, "evaluates": "head only (single forward; KV not exercised)"})
    # --- incremental (KV configurations)
    for f in sorted(x for x in files if x.startswith("incremental")):
        d = json.load(open(files[f], encoding="utf-8"))
        ref = d.get("single_forward_bf16_head", {})
        base = d.get("baseline", {})
        for cfg in ("baseline", "A", "B", "B1", "C", "C0"):
            r = d.get(cfg)
            if not r:
                continue
            quality.append({"source": f, "pilot": True, "config": cfg, "description": CONFIG_DESC.get(cfg, ""), "task": f"wikitext2 incremental ctx{r['ctx']} prefill{r['prefill']}",
                            "metric": "ppl", "n_samples": r["tokens_scored"], "baseline": base.get("ppl"), "value": r["ppl"],
                            "abs_diff": r["ppl"] - base["ppl"] if base else None, "rel_diff_pct": 100 * (r["ppl"] - base["ppl"]) / base["ppl"] if base else None,
                            "pp_diff": None, "stderr": None, "evaluates": "head + real incremental KV cache (teacher-forced)",
                            "single_forward_ref_ppl": ref.get("ppl"), "nll_by_position_bucket": json.dumps(r.get("nll_by_position_bucket"))})
    write_csv(os.path.join(RESULTS_DIR, "quality_comparison.csv"), quality)

    # --- memory
    memrows = []
    if "memory.json" in files:
        mem = read_json("memory.json")
        for cfg in ("baseline", "A", "B", "B1", "C", "C0"):
            m = mem.get(cfg)
            if not m:
                continue
            for r in m["sweep"]:
                if "ctx" not in r:
                    continue
                memrows.append({"config": cfg, "description": CONFIG_DESC[cfg], "head_bytes_resident": m["head_bytes"], **r})
    if "head_int4.json" in files:
        h = read_json("head_int4.json")
        memrows.append({"config": "head_int4_detail", "description": h["method"], "original_bytes": h["original_bytes"], "logical_total_bytes": h["total_bytes"],
                        "logical_metadata_bytes": h["metadata_bytes"], "effective_bits_per_element": h["effective_bits_per_element"],
                        "resident_bytes_as_stored": h["total_resident_bytes"], "dequant_temporary_bytes_per_forward": h["dequant_temporary_bytes_per_forward"],
                        "rel_fro_error": h["rel_fro_error"], "max_abs_error": h["max_abs_error"]})
    write_csv(os.path.join(RESULTS_DIR, "memory_comparison.csv"), memrows)

    # --- latency
    latrows = []
    if "latency.json" in files:
        lat = read_json("latency.json")
        for cfg in ("baseline", "A", "B", "B1", "C", "C0"):
            l = lat.get(cfg)
            if not l:
                continue
            for k, v in l.items():
                if k.startswith("prompt"):
                    latrows.append({"config": cfg, "description": CONFIG_DESC[cfg], **{kk: vv for kk, vv in v.items() if kk != "peak_cuda"},
                                    "peak_max_allocated_bytes": v["peak_cuda"].get("max_allocated_bytes")})
    write_csv(os.path.join(RESULTS_DIR, "latency_comparison.csv"), latrows)

    # --- candidate format
    audit = read_json("audit.json") if "audit.json" in files else {}
    head = read_json("head_int4.json") if "head_int4.json" in files else {}
    qint = audit.get("quantized_cache_layer0_internals", {})
    L = ["# Candidate numerical representation (EXP-0004) — CANDIDATE FORMAT, not FlexTern v1", "",
         "Status labels: MEASURED (observed on this machine with this software), INFERRED (derived), HYPOTHESIZED (not demonstrated), UNRESOLVED.", "",
         "## Ternary transformer weights (unchanged, MEASURED in EXP-0002B)", "",
         "- Packing: 2-bit codes, code = q + 1, four per byte along the output dimension (transformers BitNet packing).",
         "- Accumulation: INT8 activations × ternary; 19–22 signed bits at K = 2048–13312 (INFERRED from format).",
         "- Scale: one BF16 scalar per tensor, applied after accumulation (stored as 1/β for Falcon-E).", "",
         "## INT4 output head candidate", ""]
    if head:
        L += [f"- Method: {head['method']} (MEASURED).",
              f"- Group size {head['group_size']}; {head['groups']:,} groups; logical {head['total_bytes']/1e6:.1f} MB = {head['effective_bits_per_element']:.3f} bits/element incl. metadata ({head['metadata_bytes']/1e6:.1f} MB); stored here as {head['total_resident_bytes']/1e6:.1f} MB (zero kept in uint8).",
              f"- Weight reconstruction error: relative Frobenius {head['rel_fro_error']:.4f}, max abs {head['max_abs_error']:.4g} (MEASURED).",
              f"- Execution requirement: this runtime dequantizes to BF16 ({head['dequant_temporary_bytes_per_forward']/1e6:.1f} MB temporary) every forward and runs a BF16 matmul. A hardware consumer would need an INT4-weight × activation path with per-group scale/zero (HYPOTHESIZED).",
              "- Accuracy impact: see quality_comparison.csv (lm_eval tasks and incremental perplexity), MEASURED."]
    L += ["", "## int4 KV-cache candidate", ""]
    if qint:
        qk = qint.get("_quantized_keys") or {}
        L += [f"- Implementation: transformers `QuantizedCache`, {audit.get('quantized_cache_settings', {}).get('backend')} backend, {qint.get('nbits')}-bit, group {qint.get('q_group_size')}, residual {qint.get('residual_length')} tokens kept in BF16 (MEASURED).",
              f"- Stored tensors (layer 0 after a ~400-token generation): data {qk.get('_data', {}).get('dtype')} {qk.get('_data', {}).get('shape')}, scale {qk.get('_scale', {}).get('dtype')} {qk.get('_scale', {}).get('shape')}, zero {qk.get('_zero', {}).get('dtype')} {qk.get('_zero', {}).get('shape')}; effective {qk.get('effective_bits_per_element', float('nan')):.2f} bits/element for the quantized part (MEASURED).",
              f"- Cache organization: lazy init quantizes the prefill; new tokens accumulate in a BF16 residual; when it reaches {qint.get('residual_length')} tokens the whole history (dequantized old + residual + new) is re-quantized (MEASURED from source and exercised).",
              "- Decode requirement on this runtime: the entire layer cache is dequantized into a BF16 temporary before every attention call, so resident memory shrinks but the bytes read by attention do not (MEASURED behaviour). A hardware consumer would need attention that reads 4-bit K/V with per-group scale/zero directly (HYPOTHESIZED).",
              "- Update requirement: periodic re-quantization of the history; whether that is idempotent depends on grouping (see test_requantization_drift and the per-position NLL buckets) — MEASURED in this experiment.",
              "- Accuracy impact: incremental perplexity vs the BF16-cache baseline, MEASURED; generation samples, qualitative."]
    L += ["", "## What is UNRESOLVED", "",
          "- Whether an INT4-weight head path or a 4-bit-KV attention path can be built without the dequantization temporaries (no kernel measured).",
          "- Whether the quality results transfer to other checkpoints (one model evaluated).",
          "- Acceptable-loss threshold (founder decision; not predeclared).",
          "- Benefit attribution: both representations run on any GPU (category A); hardware-specific benefit (B/C) is unresolved."]
    with open(os.path.join(RESULTS_DIR, "candidate_format.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"quality rows {len(quality)}, memory rows {len(memrows)}, latency rows {len(latrows)}; wrote candidate_format.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
