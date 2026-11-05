"""EXP-0011 Stage 1 analysis: thread selection, quality matching, ratios and the preregistered speed/memory outcome.

Pure functions (tested in tests/test_ternary_vs_int4.py) plus a CLI that reads results/ternary_vs_int4/{quality,perf} and writes
results/ternary_vs_int4/{perf_models,thread_choice,summary}.json. The outcome labels are about SPEED AND MEMORY ONLY. They are not
energy results and say nothing about FlexTern hardware.

Decision structure (locked in protocol_v1.json):
  * the matched pool = INT4 models with quality macro >= ternary macro - tolerance; an EMPTY pool yields NO_MATCHED_COMPARATOR and no ratios
    (a quality-vs-speed table of every timed model is still produced);
  * headline comparator is chosen per metric per cell (the fastest pool member: conservative for ternary); the same ratios are also reported
    against one fixed primary comparator;
  * the decode audit flag compares the decode speed-up with the bandwidth-implied ratio from bytes_model (weights streamed + KV at depth).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from . import config as C
from . import perf as P

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "ternary_vs_int4"


def geomean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def macro(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)


def matched_set(macro_by_model: dict[str, float], tol: float) -> list[str]:
    """INT4 models whose macro accuracy >= the ternary model's minus tol (i.e. at least as good as the ternary within tolerance). May be empty."""
    q = macro_by_model[C.TERNARY]
    return [m for m in C.INT4 if m in macro_by_model and macro_by_model[m] >= q - tol]


def metrics(ttft_rec: dict, total_rec: dict) -> dict:
    d = P.derive(ttft_rec["samples_ns"], total_rec["samples_ns"], ttft_rec["n_prompt"], total_rec["n_gen"])
    d["rss_kb"] = total_rec.get("max_rss_kb")
    d["spread_total"] = (max(total_rec["samples_ns"]) - min(total_rec["samples_ns"])) / statistics.median(total_rec["samples_ns"])
    return d


def cell_ratios(tern: dict, cands: list[dict]) -> dict:
    """tern/cands are metric dicts (ttft_s, total_s, decode_tps, rss_kb, id). The comparator is chosen per metric as the best conventional
    alternative (the fastest matched INT4), conservative for the ternary model. Ratios >1 favour ternary."""
    best_dec = max(cands, key=lambda c: c["decode_tps"])
    best_ttft = min(cands, key=lambda c: c["ttft_s"])
    best_tot = min(cands, key=lambda c: c["total_s"])
    return {"decode": tern["decode_tps"] / best_dec["decode_tps"], "ttft": best_ttft["ttft_s"] / tern["ttft_s"],
            "total": best_tot["total_s"] / tern["total_s"],
            "memory": (best_tot["rss_kb"] / tern["rss_kb"]) if tern.get("rss_kb") and best_tot.get("rss_kb") else None,
            "decode_comparator": best_dec["id"], "ttft_comparator": best_ttft["id"], "total_comparator": best_tot["id"]}


def classify(ratios: list[dict], tol: float, q_tern: float, q_by_model: dict[str, float]) -> dict:
    """Preregistered Stage 1 outcome. Each ratio dict needs total/decode/ttft, bytes_ratio (bandwidth-implied decode ratio against the decode
    comparator) and total_comparator (id 'MODEL|runtime|tN'). Headline = geometric mean over cells of the end-to-end latency ratio."""
    g_tot = geomean(r["total"] for r in ratios)
    g_dec = geomean(r["decode"] for r in ratios)
    g_ttft = geomean(r["ttft"] for r in ratios)
    g_bytes = geomean(r["bytes_ratio"] for r in ratios)
    worst_tot = min(r["total"] for r in ratios)
    label = "SPEED_PASS" if g_tot >= C.SPEED_PASS else "SPEED_MIDDLE" if g_tot >= C.SPEED_MIDDLE else "SPEED_LOW"
    flags = []
    if g_dec >= C.SPEED_MIDDLE and g_ttft < C.TTFT_PARITY:
        flags.append("MIXED_DECODE_WIN_PREFILL_LOSS")
    if g_dec > g_bytes * C.BYTES_CEILING_MARGIN:
        flags.append("AUDIT_DECODE_EXCEEDS_BANDWIDTH_IMPLIED_RATIO")
    if worst_tot < C.SPEED_MIDDLE <= g_tot:
        flags.append("WIN_NOT_UNIFORM_ACROSS_CELLS")
    gaps = [q_by_model[r["total_comparator"].split("|")[0]] - q_tern for r in ratios]
    if max(gaps) > tol:
        flags.append("COMPARATOR_ABOVE_TOLERANCE_TERNARY_DISADVANTAGED")
    return {"label": label, "geomean_total": g_tot, "geomean_decode": g_dec, "geomean_ttft": g_ttft, "geomean_bandwidth_implied_decode": g_bytes,
            "worst_cell_total": worst_tot, "flags": flags}


def no_matched() -> dict:
    return {"label": "NO_MATCHED_COMPARATOR", "flags": [], "note": "no INT4 model reaches the ternary model's quality within tolerance: no efficiency ratio is "
            "computed; see the quality-vs-speed table"}


# ---------------------------------------------------------------------------------------------- CLI


def load_quality(tag: str = "primary") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for f in sorted((RES / "quality" / tag).glob("*.json")):
        r = json.loads(f.read_text())
        acc = r.get("acc")
        if acc is None and r.get("stdout_tail") and r.get("returncode") == 0:
            try:                                   # reparse a run whose score was printed in a format the first parser did not know (result file untouched)
                from . import quality as Q
                acc = Q.parse_score(r["task"], r["stdout_tail"])["acc"]
            except ValueError:
                acc = None
        if acc is not None:
            out.setdefault(r["model"], {})[r["task"]] = acc
    return out


def quality_macros(qual) -> dict[str, float]:
    return {m: macro(s) for m, s in qual.items() if len(s) == len(C.QUALITY_TASKS)}


def choose_threads() -> dict:
    """argmin of median end-to-end latency at each regime's sweep cell, per (model, runtime, regime). Written before the main run."""
    best: dict[str, dict] = {}
    for f in (RES / "perf" / "sweep").glob("*__total.json"):
        r = json.loads(f.read_text())
        if r["returncode"] != 0:
            continue
        reg = next((k for k, (p, g) in C.THREAD_SWEEP_CELLS.items() if (p, g) == (r["p"], r["g"])), None)
        if reg is None:
            continue
        d = best.setdefault(f"{r['model']}|{r['runtime']}|{reg}", {"by_threads": {}})
        d["by_threads"][r["threads"]] = statistics.median(r["samples_ns"]) / 1e9
    for d in best.values():
        d["best_threads"] = min(d["by_threads"], key=d["by_threads"].get)
    return best


def load_main() -> dict:
    """{(model, runtime, p, g): metrics} for the main stage."""
    out = {}
    for f in (RES / "perf" / "main").glob("*__total.json"):
        tot = json.loads(f.read_text())
        ttf = f.with_name(f.name.replace("__total", "__ttft"))
        if tot["returncode"] != 0 or not ttf.exists():
            continue
        tt = json.loads(ttf.read_text())
        if tt["returncode"] != 0:
            continue
        m = metrics(tt, tot)
        m["id"] = f"{tot['model']}|{tot['runtime']}|t{tot['threads']}"
        out[(tot["model"], tot["runtime"], tot["p"], tot["g"])] = m
    return out


def summarize() -> dict:
    from . import bytes_model as BM
    qual = load_quality()
    qmacro = quality_macros(qual)
    main = load_main()
    res = {"quality_macro": qmacro, "quality_tasks": qual, "tolerances": {}}
    if C.TERNARY not in qmacro:
        res["error"] = "ternary quality missing"
        return res
    profs = {m: BM.profile(m) for m in C.MODELS}
    res["bytes_profiles"] = profs
    # quality-vs-speed table: every timed INT4 model against ternary, no matching claim
    table = {}
    for mid in {k[0] for k in main if k[0] != C.TERNARY}:
        rs = []
        for g in C.OUTPUTS:
            for pn in C.PROMPTS:
                p = C.p_eff(pn, g)
                tk = [v for k, v in main.items() if k[0] == C.TERNARY and k[2] == p and k[3] == g]
                ck = [v for k, v in main.items() if k[0] == mid and k[2] == p and k[3] == g]
                if tk and ck:
                    rs.append(cell_ratios(tk[0], ck))
        if rs:
            table[mid] = {"quality_macro": qmacro.get(mid), "quality_minus_ternary": (qmacro.get(mid, float("nan")) - qmacro[C.TERNARY]),
                          "geomean_decode_ratio": geomean(r["decode"] for r in rs), "geomean_ttft_ratio": geomean(r["ttft"] for r in rs),
                          "geomean_total_ratio": geomean(r["total"] for r in rs)}
    res["quality_vs_speed_vs_ternary"] = table
    for tol in (C.QUALITY_TOLERANCE_POINTS, *C.QUALITY_TOLERANCE_SENSITIVITY):
        matched = matched_set(qmacro, tol)
        entry = {"matched_int4": matched}
        if not matched:
            entry["outcome"] = no_matched()
            res["tolerances"][str(tol)] = entry
            continue
        ratios = []
        for g in C.OUTPUTS:
            for pn in C.PROMPTS:
                p = C.p_eff(pn, g)
                tk = [v for k, v in main.items() if k[0] == C.TERNARY and k[2] == p and k[3] == g]
                cands = [v for k, v in main.items() if k[0] in matched and k[2] == p and k[3] == g]
                if not tk or not cands:
                    continue
                r = cell_ratios(tk[0], cands)
                dec_model = r["decode_comparator"].split("|")[0]
                r["bytes_ratio"] = BM.bytes_ratio(profs[dec_model], profs[C.TERNARY], p + g / 2)
                r.update(p=p, g=g)
                prim = [v for k, v in main.items() if k[0] == C.PRIMARY_COMPARATOR and k[2] == p and k[3] == g]
                if prim:
                    pr = cell_ratios(tk[0], prim)
                    r["vs_primary"] = {k: pr[k] for k in ("decode", "ttft", "total", "memory")}
                ratios.append(r)
        if ratios:
            entry["cells"] = ratios
            entry["outcome"] = classify(ratios, tol, qmacro[C.TERNARY], qmacro)
            entry["vs_primary_geomean"] = {k: geomean(r["vs_primary"][k] for r in ratios if "vs_primary" in r) for k in ("decode", "ttft", "total")}
        res["tolerances"][str(tol)] = entry
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select", "threads", "summary"])
    a = ap.parse_args(argv)
    RES.mkdir(parents=True, exist_ok=True)
    if a.what == "select":
        qmacro = quality_macros(load_quality())
        pool = matched_set(qmacro, C.PERF_SET_TOLERANCE)
        # an empty pool still times every INT4 model so the quality-vs-speed trade-off can be reported; it yields no efficiency ratio
        models = [C.TERNARY, *(pool or [m for m in C.INT4 if m in qmacro])]
        if C.PERF_SET_OVERRIDE:
            models = [m for m in models if m in C.PERF_SET_OVERRIDE]
        out = {"quality_macro": qmacro, "tolerance": C.PERF_SET_TOLERANCE, "pool_empty": not pool, "models": models}
        (RES / "perf_models.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    elif a.what == "threads":
        out = choose_threads()
        (RES / "thread_choice.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v["best_threads"] for k, v in out.items()}, indent=1))
    else:
        out = summarize()
        (RES / "summary.json").write_text(json.dumps(out, indent=1, default=str))
        for tol, d in out.get("tolerances", {}).items():
            print(tol, d.get("outcome"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
