"""EXP-0011 Stage 1 results report: tables with run-to-run ranges, ratios against every INT4 comparator, anomaly flags, and the locked classification.
Reads results/ternary_vs_int4/perf/main; writes results/ternary_vs_int4/report.md and report.json. Speed/memory only; no energy."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from . import config as C
from . import bytes_model as BM
from . import summarize as S

RES = S.RES
SHORT = {"BITNET_2B4T_I2S": "BitNet 2B4T (ternary)", "QWEN25_3B_Q4KM": "Qwen2.5-3B Q4_K_M", "SMOLLM2_1P7B_Q4KM": "SmolLM2-1.7B Q4_K_M",
         "QWEN25_1P5B_Q4KM": "Qwen2.5-1.5B Q4_K_M", "QWEN25_1P5B_Q40": "Qwen2.5-1.5B Q4_0"}


def load_records() -> list[dict]:
    recs = []
    for f in sorted((RES / "perf" / "main").glob("*.json")):
        r = json.loads(f.read_text())
        r["_file"] = f.name
        recs.append(r)
    return recs


def flags_for(r: dict) -> list[str]:
    fl = []
    if r["returncode"] != 0:
        fl.append("FAILED")
        return fl
    s = r["samples_ns"]
    med = statistics.median(s)
    if (max(s) - min(s)) / med > 0.25:
        fl.append(f"SPREAD>{25}%")
    if max(s) > 2.0 * med:
        fl.append("STALL_SAMPLE")
    pa = r.get("power_after")
    if pa is not None and not pa.get("ok", True):
        fl.append("POWER_AFTER_NOT_OK")
    if r.get("waited_for_ac_s", 0) > 0:
        fl.append(f"WAITED_AC_{r['waited_for_ac_s']}s")
    return fl


def cell_table(main: dict, recs: list[dict]) -> tuple[list[str], list[dict]]:
    """Per (model, runtime, p, g): TTFT, prefill tps, decode tps, total, peak RSS, with ranges across the repetitions."""
    by = {}
    for r in recs:
        by[(r["model"], r["runtime"], r["p"], r["g"], r["kind"])] = r
    rows, out = [], []
    keys = sorted({(m, rt, p, g) for (m, rt, p, g, _k) in by})
    for (m, rt, p, g) in keys:
        a, b = by.get((m, rt, p, g, "ttft")), by.get((m, rt, p, g, "total"))
        if not a or not b or a["returncode"] or b["returncode"]:
            continue
        t1 = [x / 1e9 for x in a["samples_ns"]]
        tt = [x / 1e9 for x in b["samples_ns"]]
        t1m, ttm = statistics.median(t1), statistics.median(tt)
        dec = (g - 1) / max(ttm - t1m, 1e-9)
        rec = {"model": m, "runtime": rt, "threads": b["threads"], "p": p, "g": g, "ttft_s": t1m, "ttft_range": [min(t1), max(t1)],
               "prefill_tps": p / t1m, "decode_tps": dec, "total_s": ttm, "total_range": [min(tt), max(tt)], "rss_gb": b["max_rss_kb"] / 1e6,
               "flags": flags_for(a) + flags_for(b)}
        out.append(rec)
        rows.append(f"| {SHORT[m]} / {rt} t{b['threads']} | {p} | {g} | {t1m:.2f} ({min(t1):.2f}-{max(t1):.2f}) | {rec['prefill_tps']:.0f} | {dec:.1f} | "
                    f"{ttm:.2f} ({min(tt):.2f}-{max(tt):.2f}) | {rec['rss_gb']:.2f} | {', '.join(rec['flags']) or '-'} |")
    return rows, out


def main():
    recs = load_records()
    main_metrics = S.load_main()
    rows, cells = cell_table(main_metrics, recs)
    md = ["# EXP-0011 Stage 1 timing report (auto-generated; speed/memory only, no energy)", "",
          f"Invocations: {len(recs)}; failed: {sum(1 for r in recs if r['returncode'])}; flagged: {sum(1 for r in recs if flags_for(r))}.", "",
          "## Per-cell measurements (median of 3 repetitions; ranges are min-max; RSS is peak resident set of the process)", "",
          "| model / runtime / threads | prompt | output | TTFT s | prefill tok/s | decode tok/s | total s | peak RSS GB | flags |", "|---|---|---|---|---|---|---|---|---|", *rows, ""]
    summ = S.summarize()
    (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    # ratios vs every INT4 comparator, per cell, then geometric means
    md += ["## Ratios against each INT4 comparator (INT4 time / BitNet time; >1 = BitNet faster)", "",
           "| comparator | geomean TTFT | geomean decode | geomean total | min total (cell) | max total (cell) |", "|---|---|---|---|---|---|"]
    pair = {}
    profs = {m: BM.profile(m) for m in C.MODELS}
    for mid in C.INT4:
        for rt in ("upstream", "bitnet_old"):
            rs = []
            for g in C.OUTPUTS:
                for pn in C.PROMPTS:
                    p = C.p_eff(pn, g)
                    t = [v for k, v in main_metrics.items() if k[0] == C.TERNARY and k[2] == p and k[3] == g]
                    c = [v for k, v in main_metrics.items() if k[0] == mid and k[1] == rt and k[2] == p and k[3] == g]
                    if t and c:
                        rs.append((p, g, c[0]["ttft_s"] / t[0]["ttft_s"], t[0]["decode_tps"] / c[0]["decode_tps"], c[0]["total_s"] / t[0]["total_s"],
                                   (c[0]["rss_kb"] / t[0]["rss_kb"]) if c[0].get("rss_kb") and t[0].get("rss_kb") else float('nan'),
                                   BM.bytes_ratio(profs[mid], profs[C.TERNARY], p + g / 2)))
            if len(rs) == 8:
                pair[(mid, rt)] = rs
                gm = lambda i: S.geomean(x[i] for x in rs)  # noqa: E731
                lo, hi = min(rs, key=lambda x: x[4]), max(rs, key=lambda x: x[4])
                md.append(f"| {SHORT[mid]} / {rt} | {gm(2):.2f} | {gm(3):.2f} | {gm(4):.2f} | {lo[4]:.2f} (P{lo[0]},G{lo[1]}) | {hi[4]:.2f} (P{hi[0]},G{hi[1]}) |")
    md += ["", "## Per-cell total-latency ratio (INT4 total / BitNet total) by context and output length", "",
           "| comparator | G | P=512 | P=1024 | P=2048 | P=3968/3840 |", "|---|---|---|---|---|---|"]
    for (mid, rt), rs in pair.items():
        for g in C.OUTPUTS:
            vals = {x[0]: x[4] for x in rs if x[1] == g}
            md.append(f"| {SHORT[mid]} / {rt} | {g} | " + " | ".join(f"{vals[p]:.2f}" for p in sorted(vals)) + " |")
    md += ["", "## Peak resident memory and bandwidth-implied decode ratio", "",
           "| comparator | RSS ratio (INT4 / BitNet) geomean [min-max] | measured decode ratio (BitNet / INT4) geomean | bandwidth-implied decode ratio (INT4 bytes / BitNet bytes, weights + f16 KV) geomean |", "|---|---|---|---|"]
    for (mid, rt), rs in pair.items():
        rss = [x[5] for x in rs]
        md.append(f"| {SHORT[mid]} / {rt} | {S.geomean(rss):.2f} [{min(rss):.2f}-{max(rss):.2f}] | {S.geomean(x[3] for x in rs):.2f} | {S.geomean(x[6] for x in rs):.2f} |")
    md += ["", "(Decode throughput is derived as (G-1)/(total - TTFT) from two separate runs, so it is noisier than TTFT and total latency; use it with the ranges above.)"]
    md += ["", "## Preregistered classification (locked rule)", ""]
    for tol, d in summ.get("tolerances", {}).items():
        md.append(f"- tolerance {tol}: matched INT4 = {d.get('matched_int4')}; outcome = `{json.dumps(d.get('outcome'), default=str)}`")
        if d.get("vs_primary_geomean"):
            md.append(f"  - fixed primary comparator ({C.PRIMARY_COMPARATOR}) geomeans: {json.dumps({k: round(v, 3) for k, v in d['vs_primary_geomean'].items()})}")
    rep = RES / "perf" / "flag_repeats"
    if rep.exists():
        md += ["", "## Re-measurement of the flagged cells (originals are kept; the classification above uses the ORIGINAL medians)", "",
               "| cell | original total s (samples) | repeat total s (6 samples) | original flags |", "|---|---|---|---|"]
        alt = {}
        for f in sorted(rep.glob("*.json")):
            r = json.loads(f.read_text())
            if not r["returncode"]:
                alt.setdefault((r["model"], r["runtime"], r["p"], r["g"]), []).extend(r["samples_ns"])
        for (m, rt, p, g), smp in alt.items():
            orig = [x for x in recs if (x["model"], x["runtime"], x["p"], x["g"], x["kind"]) == (m, rt, p, g, "total")][0]
            md.append(f"| {SHORT[m]} / {rt} P{p} G{g} | {statistics.median(orig['samples_ns']) / 1e9:.1f} ({', '.join(f'{x / 1e9:.1f}' for x in orig['samples_ns'])}) | "
                      f"{statistics.median(smp) / 1e9:.1f} ({', '.join(f'{x / 1e9:.1f}' for x in smp)}) | {', '.join(flags_for(orig)) or '-'} |")
        mm = {k: dict(v) for k, v in main_metrics.items()}
        for (m, rt, p, g), smp in alt.items():
            for k, v in mm.items():
                if k[0] == m and k[1] == rt and k[2] == p and k[3] == g:
                    v["total_s"] = statistics.median(smp) / 1e9
        q = S.quality_macros(S.load_quality())
        for tol in (C.QUALITY_TOLERANCE_POINTS, 5.0):
            pool = S.matched_set(q, tol)
            rr = []
            for g in C.OUTPUTS:
                for pn in C.PROMPTS:
                    p = C.p_eff(pn, g)
                    t = [v for k, v in mm.items() if k[0] == C.TERNARY and k[2] == p and k[3] == g]
                    cs = [v for k, v in mm.items() if k[0] in pool and k[2] == p and k[3] == g]
                    if t and cs:
                        rr.append(S.cell_ratios(t[0], cs)["total"])
            if rr:
                md.append(f"\nSensitivity (flagged cells replaced by their repeat medians), tolerance {tol}: geomean end-to-end ratio = {S.geomean(rr):.3f}.")
    (RES / "report.md").write_text("\n".join(md))
    (RES / "report.json").write_text(json.dumps({"cells": cells}, indent=1))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
