"""EXP-0011 Stage 1 post-hoc analysis (labelled INFERRED where it interprets): runtime-independent footprint, effective memory bandwidth,
prefill throughput per body parameter, the contaminated-cell sensitivity and a bootstrap range on the headline ratio.
Writes results/ternary_vs_int4/attribution.md. Nothing here changes the locked classification."""
from __future__ import annotations

import json
import random
import statistics
import sys

from . import bytes_model as BM
from . import config as C
from . import report as R
from . import summarize as S

RES = S.RES
KV_DEPTHS = (0, 2048, 4096)


def body_params(model_id: str) -> int:
    """Elements in the 2-D weight matrices of the transformer blocks (what a prefill token multiplies by); excludes embedding and output head."""
    r = BM._reader(C.MODELS_DIR / C.MODELS[model_id]["file"])
    return sum(int(t.n_elements) for t in r.tensors if t.name.startswith("blk.") and t.name.endswith(".weight") and len(t.shape) == 2)


def bootstrap_headline(recs, pool, n=2000, seed=1):
    """Resample each invocation's repetitions with replacement, recompute medians, per-cell fastest-pool-member ratios and their geomean."""
    rng = random.Random(seed)
    by = {}
    for r in recs:
        if r["returncode"] == 0:
            by[(r["model"], r["runtime"], r["p"], r["g"], r["kind"])] = [x / 1e9 for x in r["samples_ns"]]
    cfgs = sorted({(m, rt) for (m, rt, *_x) in by})
    out = []
    for _ in range(n):
        med = {k: statistics.median([rng.choice(v) for _ in v]) for k, v in by.items()}
        cells = []
        for g in C.OUTPUTS:
            for pn in C.PROMPTS:
                p = C.p_eff(pn, g)
                t = med.get((C.TERNARY, "bitnet_old", p, g, "total"))
                cs = [med[(m, rt, p, g, "total")] for (m, rt) in cfgs if m in pool and (m, rt, p, g, "total") in med]
                if t and cs:
                    cells.append(min(cs) / t)
        out.append(S.geomean(cells))
    out.sort()
    return out[int(0.025 * n)], statistics.median(out), out[int(0.975 * n)]


def main():
    recs = R.load_records()
    profs = {m: BM.profile(m) for m in C.MODELS}
    md = ["# EXP-0011 Stage 1 attribution analysis (post-hoc; INFERRED unless stated)", ""]

    md += ["## A. Runtime-independent footprint (MEASURED from GGUF tensors; f16 KV cache)", "",
           "| model | all tensors GB (resident once loaded) | + KV at 0 | + KV at 2,048 | + KV at 4,096 | weights streamed per token GB |", "|---|---|---|---|---|---|"]
    for m in (C.TERNARY, "QWEN25_3B_Q4KM", "SMOLLM2_1P7B_Q4KM", "QWEN25_1P5B_Q4KM", "QWEN25_1P5B_Q40"):
        p = profs[m]
        row = [f"{(p['sum_all_tensor_bytes'] + p['kv_bytes_per_token'] * d) / 1e9:.2f}" for d in KV_DEPTHS]
        md.append(f"| {R.SHORT[m]} | {p['sum_all_tensor_bytes'] / 1e9:.2f} | " + " | ".join(row[:3]) + f" | {p['weight_bytes_streamed'] / 1e9:.2f} |")
    md += ["", "Static footprint (all tensors + KV) is runtime-independent; measured peak RSS (report.md) additionally depends on the runtime (the same Qwen2.5-1.5B Q4_K_M file "
           "peaks at 1.22-1.32 GB under bitnet.cpp 765741d and 1.85-1.95 GB under modern llama.cpp). Against the 3B model BitNet's footprint advantage is real "
           "(1.18 vs 1.93 GB static); against SmolLM2 it holds only at long context (SmolLM2 has 32 KV heads); against the 1.5B models it does not exist.", ""]

    md += ["## B. Effective memory bandwidth in decode (INFERRED): decode tok/s x bytes streamed per token (weights + f16 KV at depth P+G/2), GB/s", "",
           "| model / runtime | min | median | max over the 8 cells |", "|---|---|---|---|"]
    cells = json.loads((RES / "report.json").read_text())["cells"]
    for key in sorted({(c["model"], c["runtime"]) for c in cells}):
        bw = [c["decode_tps"] * BM.decode_bytes(profs[c["model"]], c["p"] + c["g"] / 2) / 1e9 for c in cells if (c["model"], c["runtime"]) == key]
        md.append(f"| {R.SHORT[key[0]]} / {key[1]} | {min(bw):.1f} | {statistics.median(bw):.1f} | {max(bw):.1f} |")
    md += ["", "(Derived decode throughput is noisy; the P=3840,G=256 BitNet cell is affected by an AC-adapter drop in the original run.)", ""]

    md += ["## C. Prefill throughput per body parameter (INFERRED): prefill tok/s x block-matrix parameters (G param-tokens/s), by prompt length", "",
           "| model / runtime | P=512 | P=1024 | P=2048 | P~3.9k | body params (B) |", "|---|---|---|---|---|---|"]
    for key in sorted({(c["model"], c["runtime"]) for c in cells}):
        bp = body_params(key[0])
        vals = {}
        for c in cells:
            if (c["model"], c["runtime"]) == key and c["g"] == 128:
                vals[c["p"]] = c["prefill_tps"] * bp / 1e9
        md.append(f"| {R.SHORT[key[0]]} / {key[1]} | " + " | ".join(f"{vals[p]:.0f}" for p in sorted(vals)) + f" | {bp / 1e9:.2f} |")
    md += ["", "If these throughputs are similar across models on the same runtime, prefill time scales with body size rather than with weight format.", ""]

    md += ["## D. Sensitivity to the contaminated cell (BitNet P3840,G256; AC adapter dropped during the original run)", ""]
    rep = RES / "perf" / "flag_repeats"
    alt = {}
    for f in sorted(rep.glob("*.json")):
        r = json.loads(f.read_text())
        if not r["returncode"]:
            alt.setdefault((r["model"], r["runtime"], r["p"], r["g"]), []).extend(r["samples_ns"])
    mm = {k: dict(v) for k, v in S.load_main().items()}
    for (m, rt, p, g), smp in alt.items():
        mm[(m, rt, p, g)]["total_s"] = statistics.median(smp) / 1e9
    md += ["| comparator | context (G=128): P=512 / 1024 / 2048 / 3968 | context (G=256), repeat-substituted: P=512 / 1024 / 2048 / 3840 |", "|---|---|---|"]
    for mid in ("QWEN25_3B_Q4KM", "SMOLLM2_1P7B_Q4KM", "QWEN25_1P5B_Q4KM"):
        row = []
        for g in C.OUTPUTS:
            vals = []
            for pn in C.PROMPTS:
                p = C.p_eff(pn, g)
                t = mm[(C.TERNARY, "bitnet_old", p, g)]["total_s"]
                c = mm[(mid, "upstream", p, g)]["total_s"]
                vals.append(c / t)
            row.append(" / ".join(f"{v:.2f}" for v in vals))
        md.append(f"| {R.SHORT[mid]} / upstream | {row[0]} | {row[1]} |")
    md += ["", "The G=256 column's last entry uses the repeat-measured values for both sides where available; the context trend against the 3B model is carried by the G=128 column, "
           "whose last cell (P=3968) was not flagged.", ""]

    q = S.quality_macros(S.load_quality())
    md += ["## E. Uncertainty on the headline (bootstrap of repetitions, 2,000 draws, seed 1; medians recomputed; fastest pool member per cell)", ""]
    for tol in (C.QUALITY_TOLERANCE_POINTS, 5.0):
        pool = S.matched_set(q, tol)
        lo, mid, hi = bootstrap_headline(recs, pool)
        md.append(f"- tolerance {tol} (pool {pool}): geometric-mean end-to-end ratio {mid:.3f}, 95% bootstrap range [{lo:.3f}, {hi:.3f}] (locked point estimate in report.md). "
                  "This range reflects only run-to-run noise in 3 repetitions per cell, not systematic error (WSL2, thermal state, runtime vintage).")

    md += ["", "## F. Head quantization, HYPOTHESIZED (not measured; quality untested)",
           "BitNet's tied 128,256 x 2,560 head is F16 (0.657 GB). A Q6_K-class head (6.5625 bits/weight) would be about 0.269 GB, lowering the weights streamed per token from 1.18 to about 0.79 GB "
           "(bandwidth-implied decode vs Qwen2.5-1.5B Q4_K_M would change from 0.78 to roughly 1.1-1.2 at short context). Whether BitNet keeps its quality with a quantized head is unknown here; "
           "EXP-0004 found a Falcon-E 1B model tolerated an INT4 head, which is relevant precedent only.", ""]
    (RES / "attribution.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
