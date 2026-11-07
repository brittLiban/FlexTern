"""Aggregate EXP-0012 result JSONs into tables and gate verdicts (reads only measured files; writes markdown + json).
Usage: python -m analysis.predictive_state.report TEST_JSON PROBE_JSON PRED_JSON LATENCY_JSON OUTDIR
Gates are exactly those in protocol_v1.json (with amendments recorded in the experiment record)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import cost_model as C

CONDS = ["NOM", "OCC", "DIST", "HARD"]
RNG = np.random.default_rng(0)


def load(path):
    return json.load(open(path)) if path and Path(path).exists() else {}


def vecs(res, arm, cond, spec="full"):
    """list over seeds of 0/1 episode vectors (picks_frac is stored as a mean; LONG uses success_vec of full completion)."""
    out = {}
    for k, v in res.items():
        a, s, split, c, sp = k.split("|")
        if a == arm and c == cond and sp == spec and split == "test":
            out[int(s)] = np.array(v["success_vec"], float)
    return out


def mean_over_seeds(v):
    return float(np.mean([x.mean() for x in v.values()])) if v else float("nan")


def macro(res, arm, spec="full"):
    ms = [mean_over_seeds(vecs(res, arm, c, spec)) for c in CONDS]
    return float(np.mean(ms)) if not any(np.isnan(ms)) else float("nan")


def macro_boot(res, arms, spec="full", n=2000, fn=None):
    """paired bootstrap over episodes within (arm-independent) seed/condition cells; returns draws of fn(macros...)"""
    cells = {}
    for a in arms:
        for c in CONDS:
            for s, v in vecs(res, a, c, spec if isinstance(spec, str) else spec[a]).items():
                cells.setdefault((c, s), {})[a] = v
    keys = [k for k, d in cells.items() if len(d) == len(arms)]
    if not keys:
        return None
    draws = []
    for _ in range(n):
        per = {a: [] for a in arms}
        for k in keys:
            m = min(len(v) for v in cells[k].values())
            idx = RNG.integers(0, m, m)
            for a in arms:
                per[a].append(cells[k][a][:m][idx].mean())
        draws.append(fn(*[np.mean(per[a]) for a in arms]))
    return np.array(draws)


def row(res, arm, spec="full"):
    cells = [mean_over_seeds(vecs(res, arm, c, spec)) for c in CONDS]
    seeds = sorted({int(k.split("|")[1]) for k in res if k.startswith(arm + "|") and k.endswith("|" + spec) and "|test|" in k})
    per_seed = []
    for s in seeds:
        ms = [vecs(res, arm, c, spec).get(s) for c in CONDS]
        per_seed.append(None if any(m is None for m in ms) else float(np.mean([m.mean() for m in ms])))
    lg = vecs(res, arm, "LONG", spec)
    pk = [v["picks_frac"] for k, v in res.items() if k.startswith(f"{arm}|") and "|LONG|" in k and k.endswith("|" + spec) and "|test|" in k]
    ef = [v["enc_frac"] for k, v in res.items() if k.startswith(f"{arm}|") and "|test|" in k and k.endswith("|" + spec) and "|LONG|" not in k]
    return {"cells": cells, "macro": float(np.nanmean(cells)), "per_seed_macro": per_seed, "long_complete": mean_over_seeds(lg),
            "long_picks_frac": float(np.mean(pk)) if pk else float("nan"), "enc_frac": float(np.mean(ef)) if ef else float("nan"), "seeds": seeds}


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    test, probe, pred, lat, out = sys.argv[1:6]
    res, pr, pd_, lt = load(test), load(probe), load(pred), load(lat)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    arms = sorted({k.split("|")[0] for k in res})
    full_arms = [a for a in arms if row(res, a)["seeds"]]
    md = ["# EXP-0012 results (TEST split, locked protocol + amendments)\n"]
    md.append("All numbers MEASURED in simulation (OccPick2D, 32x32, one laptop GPU/CPU). Success = fraction of episodes completing the task; cells NOM/OCC/DIST/HARD, macro = their mean; LONG = fraction of 10-pick episodes fully completed; picks = mean fraction of picks completed.\n")
    md.append("## Full-perception success by arm (mean over seeds)\n")
    md.append("| arm | NOM | OCC | DIST | HARD | macro | per-seed macro | LONG complete | LONG picks frac | seeds |\n|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for a in full_arms:
        r = row(res, a)
        rows[a] = r
        md.append(f"| {a} | " + " | ".join(fmt(x) for x in r["cells"]) + f" | {fmt(r['macro'])} | {', '.join(fmt(x) for x in r['per_seed_macro'])} | {fmt(r['long_complete'])} | {fmt(r['long_picks_frac'])} | {len(r['seeds'])} |")
    gates = {}
    G = rows.get("G")
    if G:
        gates["G1_floor"] = {"macro": G["macro"], "NOM": G["cells"][0], "pass": bool(G["macro"] >= 0.60 and G["cells"][0] >= 0.70)}
        for a in ("G-Tu", "G-Ta"):
            if a in rows:
                d = macro_boot(res, ["G", a], fn=lambda g, t: t / g)
                lo = float(np.percentile(d, 5)) if d is not None else float("nan")
                ratio = rows[a]["macro"] / G["macro"]
                gates[f"G2_{a}"] = {"ratio": ratio, "ratio_lower95_onesided": lo, "abs_macro": rows[a]["macro"],
                                    "relative_pass": bool(ratio >= 0.95 and lo >= 0.90), "absolute_pass": bool(rows[a]["macro"] >= 0.60),
                                    "pass": bool(ratio >= 0.95 and lo >= 0.90 and rows[a]["macro"] >= 0.60)}
            if a in rows and rows["G"]["long_complete"] > 0:
                gates[f"G5_long_{a}"] = {"picks_ratio": rows[a]["long_picks_frac"] / G["long_picks_frac"],
                                         "pass_picks": bool(rows[a]["long_picks_frac"] >= 0.95 * G["long_picks_frac"])}
    # ---- questions A / B / C tables
    md.append("\n## Question A: does persistent state help? (macro over OCC/HARD/DIST vs controls)\n")
    md.append("| arm | OCC | DIST | HARD | mean(OCC,DIST,HARD) |\n|---|---|---|---|---|")
    for a in ("R", "S4", "T16", "G0", "G"):
        if a in rows:
            c = rows[a]["cells"]
            md.append(f"| {a} | {fmt(c[1])} | {fmt(c[2])} | {fmt(c[3])} | {fmt(np.mean(c[1:]))} |")
    md.append("\n## Question B: JEPA objective (G vs G0, paired by seed)\n")
    if "G" in rows and "G0" in rows:
        d = macro_boot(res, ["G0", "G"], fn=lambda g0, g: g - g0)
        md.append(f"macro G - G0 = {fmt(rows['G']['macro'] - rows['G0']['macro'])}; bootstrap 95% [{fmt(np.percentile(d,2.5))}, {fmt(np.percentile(d,97.5))}]\n")
        gates["B_jepa_delta"] = {"delta": rows["G"]["macro"] - rows["G0"]["macro"], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    # ---- adaptive perception
    md.append("\n## Question D: adaptive perception (TEST; encoder fraction = encoder runs / control steps)\n")
    md.append("| model | trigger@budget | NOM | OCC | DIST | HARD | macro | enc fraction | macro / own full | DIST / own full |\n|---|---|---|---|---|---|---|---|---|---|")
    adapt = {}
    specs = sorted({k.split("|")[4] for k in res if k.split("|")[4] != "full"})
    for a in [x for x in arms if x.startswith("G")]:
        base = rows.get(a)
        for sp in ["full"] + specs:
            if sp != "full" and not row(res, a, sp)["seeds"]:
                continue
            r = row(res, a, sp)
            if not r["seeds"]:
                continue
            adapt[(a, sp)] = r
            md.append(f"| {a} | {sp} | " + " | ".join(fmt(x) for x in r["cells"]) + f" | {fmt(r['macro'])} | {fmt(r['enc_frac'])} | {fmt(r['macro']/base['macro']) if base else 'n/a'} | {fmt(r['cells'][2]/base['cells'][2]) if base else 'n/a'} |")
    # G3 for the ternary candidates and the FP16 control
    for a in ("G-Ta", "G-Tu", "G"):
        if (a, "learned@0.33") in adapt and (a, "full") in adapt:
            L, Fu = adapt[(a, "learned@0.33")], adapt[(a, "full")]
            red = 1 - L["enc_frac"] / Fu["enc_frac"]
            others = [adapt[(a, s)]["macro"] for s in ("fixed@0.33", "pix@0.33") if (a, s) in adapt]
            gates[f"G3_{a}"] = {"enc_reduction": red, "macro_ratio": L["macro"] / Fu["macro"], "dist_ratio": L["cells"][2] / Fu["cells"][2],
                                "learned_minus_best_control_macro": L["macro"] - max(others) if others else None,
                                "pass": bool(red >= 0.5 and L["macro"] / Fu["macro"] >= 0.95 and L["cells"][2] / Fu["cells"][2] >= 0.90
                                             and others and L["macro"] - max(others) >= 0.03)}
    md.append("\n## Exploratory (post-hoc, not gated): long blackouts (30-50 steps)\n")
    md.append("| arm | BLK40 | BLK40D (+displacement) | seeds |\n|---|---|---|---|")
    for a in full_arms:
        b1, b2 = mean_over_seeds(vecs(res, a, "BLK40")), mean_over_seeds(vecs(res, a, "BLK40D"))
        md.append(f"| {a} | {fmt(b1)} | {fmt(b2)} | {len(vecs(res, a, 'BLK40'))} |")
    # ---- G4 analytical cost
    md.append("\n## Analytical per-step cost (ANALYTICAL; weight+state bytes and MACs; not measured)\n")
    md.append("| arm | enc fraction | MACs/step (k) | bytes/step (kB) |\n|---|---|---|---|")
    cost = {}
    for a, f, trig in [("G", 1.0, False), ("G-I8a", 1.0, False), ("G-I4u", 1.0, False), ("G-I4a", 1.0, False), ("G-Tu", 1.0, False), ("G-Ta", 1.0, False), ("R", 1.0, False), ("T16", 1.0, False)]:
        c = C.step_cost(a, f, trig)
        cost[(a, f)] = c
        md.append(f"| {a} | {f:.2f} | {c['macs']/1e3:.0f} | {c['bytes']/1e3:.1f} |")
    if "G" in rows:
        eligible = [a for a in ("G-I4a", "G-I4u", "G-I8a") if a in rows and rows[a]["macro"] >= 0.95 * G["macro"]]
        md.append(f"\nControls retaining >=95% of G macro success: {eligible or 'none'}.\n")
        md.append("\n### G4 pairs: candidate vs strongest eligible control under the SAME learned trigger (measured test encoder fraction), analytical ratio control/candidate\n")
        md.append("(all INT4/INT8 controls listed; 'eligible' = retains >=95% of G macro success, the locked eligibility rule. Scope matters: G-Ta quantizes encoder+updater, G-Tu only the updater; the like-for-like INT4 control is the same scope.)\n")
        md.append("| candidate | control | eligible | enc frac cand/control | bytes ratio | MACs ratio |\n|---|---|---|---|---|---|")
        for cand in ("G-Ta", "G-Tu"):
            if (cand, "learned@0.33") not in adapt:
                continue
            fc = adapt[(cand, "learned@0.33")]["enc_frac"]
            for ctl in ("G-I4a", "G-I4u"):
                if (ctl, "learned@0.33") not in adapt:
                    continue
                fk = adapt[(ctl, "learned@0.33")]["enc_frac"]
                cc, ck = C.step_cost(cand, fc, True), C.step_cost(ctl, fk, True)
                br, mr = ck["bytes"] / cc["bytes"], ck["macs"] / cc["macs"]
                el = ctl in eligible
                md.append(f"| {cand} | {ctl} | {'yes' if el else 'no'} | {fc:.2f}/{fk:.2f} | {br:.2f} | {mr:.2f} |")
                gates[f"G4_{cand}_vs_{ctl}"] = {"eligible_control": el, "bytes_ratio": br, "macs_ratio": mr, "pass_analytical": bool(br >= 2 and mr >= 2)}
            cc = C.step_cost(cand, fc, True); cg = C.step_cost("G", 1.0, False)
            gates[f"G4_{cand}_vs_G_full"] = {"bytes_ratio": cg["bytes"] / cc["bytes"], "macs_ratio": cg["macs"] / cc["macs"]}
    # ---- probe, prediction, latency
    if pr:
        md.append("\n## State usefulness: ridge probes from the recurrent state / encoder embedding (R^2; standardized features, CV lambda; fit on DEV LONG steps<=100, scored on TEST LONG)\n")
        md.append("Primary probe = h -> position of the occluded (blackout-frame) target. h->target on valid frames and z->target are positive controls; if the primary probe is near 0 while controls are high, h does not carry the remembered position.\n")
        md.append("| arm | seed | probe | n fit | R2 early (<=100) | R2 late (>160) | n late |\n|---|---|---|---|---|---|---|")
        for k, v in sorted(pr.items()):
            for name in ("h->target|blackout", "h->target|valid", "z->target|valid", "h->goal|blackout"):
                d = v.get(name)
                if not d or "early_le100" not in d:
                    md.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {name} | {d['n_fit'] if d else 0} | n/a | n/a | n/a |")
                    continue
                md.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {name} | {d['n_fit']} | {fmt(d['early_le100']['r2'])} | {fmt(d['late_gt160']['r2'])} | {d['late_gt160']['n']} |")
        md.append("\n| arm | seed | mean abs h early/late | max abs h | NaN |\n|---|---|---|---|---|")
        for k, v in sorted(pr.items()):
            md.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {fmt(v['mean_abs_h_le100'])}/{fmt(v['mean_abs_h_gt160'])} | {fmt(v['max_abs_h'])} | {v['nan']} |")
        for a_ in ("G", "G-Tu", "G-Ta"):
            rs = [v["h->target|blackout"] for k, v in pr.items() if k.startswith(a_ + "|") and "early_le100" in v.get("h->target|blackout", {})]
            if rs and all(r["late_gt160"]["r2"] is not None and r["early_le100"]["r2"] is not None for r in rs):
                gates.setdefault("G5_probe", {})[a_] = {"early": float(np.mean([r["early_le100"]["r2"] for r in rs])), "late": float(np.mean([r["late_gt160"]["r2"] for r in rs]))}
        pg = gates.get("G5_probe", {})
        if "G" in pg:
            for a_ in ("G-Tu", "G-Ta"):
                if a_ in pg:
                    pg[a_]["late_drop_vs_G"] = pg["G"]["late"] - pg[a_]["late"]
                    pg[a_]["primary_probe_informative"] = bool(pg["G"]["early"] > 0.2)
    # per-pick conditional success (LONG drift diagnostic) and final z_std
    md.append("\n## LONG: per-pick conditional success (P(complete pick j | completed j-1)); episodes end at first failure\n")
    md.append("| arm | picks 1-2 | picks 3-5 | picks 6-10 | episodes reaching pick 6 |\n|---|---|---|---|---|")
    for a_ in full_arms:
        vv = [np.array(v["picks_done_vec"]) for k, v in res.items() if k.startswith(a_ + "|") and "|test|LONG|full" in k]
        if not vv:
            continue
        pdn = np.concatenate(vv)

        def cond(lo, hi):
            num = sum((pdn >= j + 1).sum() for j in range(lo, hi))
            den = sum((pdn >= j).sum() for j in range(lo, hi))
            return num / den if den else float("nan")
        md.append(f"| {a_} | {fmt(cond(0,2))} | {fmt(cond(2,5))} | {fmt(cond(5,10))} | {int((pdn>=5).sum())} of {len(pdn)} |")
    try:
        import os
        import torch
        ck = os.environ.get("PS_CKPT", "/home/liban/ps_ckpt")
        zs = {}
        for f in sorted(os.listdir(ck)):
            if f.endswith(".pt") and f.startswith("G") and "G0" not in f:
                lg = torch.load(os.path.join(ck, f), map_location="cpu")["log"]
                zs[f[:-3]] = lg[-1].get("z_std")
        md.append("\n## Embedding collapse monitor: final per-dimension std of z across the training batch (JEPA arms)\n")
        md.append("| run | z_std |\n|---|---|")
        for k, v in zs.items():
            md.append(f"| {k} | {fmt(v)} |")
    except Exception as e:
        md.append(f"\n(z_std unavailable: {e})")
    if pd_:
        md.append("\n## Multi-step latent prediction (cosine to EMA target embedding; k steps open-loop; DEV NOM+DIST)\n")
        md.append("| arm | seed | k=1 pred/shuffled/persist | k=4 | k=8 |\n|---|---|---|---|---|")
        for k, v in sorted(pd_.items()):
            f = lambda i: f"{v['cos_pred'][i]:.3f}/{v['cos_shuffled_action'][i]:.3f}/{v['cos_persistence'][i]:.3f}"
            md.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {f(0)} | {f(3)} | {f(7)} |")
    if lt:
        md.append("\n## Measured CPU latency (PyTorch FP32 reference, batch 1; quantized arms NOT timed)\n")
        md.append("| run | p50 ms | p95 ms | max ms | enc fraction |\n|---|---|---|---|---|")
        for k, v in lt["runs"].items():
            md.append(f"| {k} | {v['p50_ms']:.3f} | {v['p95_ms']:.3f} | {v['max_ms']:.3f} | {v['enc_frac']:.2f} |")
    md.append("\n## Gate summary (computed)\n```json\n" + json.dumps(gates, indent=1, default=float) + "\n```\n")
    (out / "report.md").write_text("\n".join(md))
    (out / "gates.json").write_text(json.dumps(gates, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
