"""EXP-0013 Phase 1.1: derive the summary tables (paired statistics, timing with rep-level uncertainty, horizon grid) from the raw result files.
Usage: python -m analysis.predictive_state.analyze_phase11 RESULTS_DIR   (writes summary.json, prints markdown tables)"""
import json
import math
import sys
from pathlib import Path

import numpy as np

T975 = {4: 2.776, 6: 2.447, 11: 2.201}      # two-sided 97.5% t quantiles by degrees of freedom (reps-1)


def mcnemar(a, b):
    """a, b per-episode 0/1 vectors on the same episodes: (a-only successes, b-only successes, exact two-sided p)."""
    a, b = np.asarray(a), np.asarray(b)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def paired_tables(d):
    out = {}
    for cn, cond in d["conds"].items():
        ep = {k: np.array(v["per_episode_success_at_budget"]) for k, v in cond.items() if not k.startswith("_")}
        out[cn] = {"success": {k: cond[k]["success_at_budget"] for k in ep},
                   "ci95": {k: cond[k]["ci95"] for k in ep},
                   "A_vs_B": mcnemar(ep["obs_A_single_frame"], ep["obs_B_map_memory"]),
                   "B_vs_oracle": mcnemar(ep["obs_B_map_memory"], ep["oracle_descent"]),
                   "A_vs_oracle": mcnemar(ep["obs_A_single_frame"], ep["oracle_descent"]),
                   "B_vs_reactive_memory": mcnemar(ep["obs_B_map_memory"], ep["reactive_memory"]),
                   "A_vs_greedy": mcnemar(ep["obs_A_single_frame"], ep["greedy"])}
    return out


def timing_tables(t):
    out = {"batch1": {}, "batched": {}, "mac_budget": t["mac_budget_per_step"]}
    tot = sum(t["mac_budget_per_step"].values())
    out["encoder_mac_fraction"] = t["mac_budget_per_step"]["enc"] / (tot - t["mac_budget_per_step"]["trig"])      # trigger head only in learned mode
    for k, v in t["batch1"].items():
        row = {"enc_frac": v["enc_frac"], "n_steps": v["n_steps"], "reps": v["reps"]}
        base = np.array(v["variants"]["legacy"]["rep_means_ms"])
        for name, s in v["variants"].items():
            r = np.array(s["rep_means_ms"])
            d = r - base
            h = T975[len(d) - 1] * d.std(ddof=1) / math.sqrt(len(d))
            row[name] = {"p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "mean_ms": float(r.mean()), "paired_rep_diff_vs_legacy_ms": float(d.mean()),
                         "paired_rep_diff_ci95_ms": [float(d.mean() - h), float(d.mean() + h)]}
        row["component_mean_ms_runner_tm"] = v["component_mean_ms_runner_tm"]
        row["theoretical_mac_saving"] = float(out["encoder_mac_fraction"] * (1 - v["enc_frac"]))
        out["batch1"][k] = row
    for k, v in t["batched"].items():
        row = {}
        base = np.array(v["variants"]["legacy"]["wall_s_all"])
        for name, s in v["variants"].items():
            r = np.array(s["wall_s_all"])
            d = r - base
            h = T975[len(d) - 1] * d.std(ddof=1) / math.sqrt(len(d))
            row[name] = {"wall_s_median": s["wall_s_median"], "paired_diff_vs_legacy_s": float(d.mean()), "paired_diff_ci95_s": [float(d.mean() - h), float(d.mean() + h)],
                         "relative_change_pct": float(100 * d.mean() / base.mean()), "enc_samples": s.get("counters", {}).get("enc_samples")}
        row["identical_per_episode_success"] = v["identical_per_episode_success"]
        out["batched"][k] = row
    return out


def horizon_table(files):
    rows = {}
    for f in files:
        h = json.load(open(f))
        for name, s in h["runs"].items():
            rows[name] = (s["success_at_budget"], s["sim_steps_per_decision"], h["n"])
    return rows


def main():
    d = Path(sys.argv[1])
    out = {"paired": paired_tables(json.load(open(d / "paired.json"))), "timing": timing_tables(json.load(open(d / "timing_paired.json")))}
    out["horizon"] = horizon_table([d / "horizon_anchor.json", d / "horizon_grid_B320.json", d / "horizon_grid_B2720_euclid.json"])
    json.dump(out, open(d / "summary.json", "w"), indent=1)
    for cn, v in out["paired"].items():
        print(cn, {k: round(x, 3) for k, x in v["success"].items()}, "A-vs-B (A only, B only, p):", v["A_vs_B"], "B-vs-oracle:", v["B_vs_oracle"])
    for k, v in out["timing"]["batch1"].items():
        print(k, "enc_frac %.2f theo-MAC-saving %.2f" % (v["enc_frac"], v["theoretical_mac_saving"]),
              {n: (round(v[n]["mean_ms"], 3), round(v[n]["paired_rep_diff_vs_legacy_ms"], 3), [round(x, 3) for x in v[n]["paired_rep_diff_ci95_ms"]]) for n in ("legacy", "runner", "runner_nt", "runner_tm")},
              {a: round(b, 3) for a, b in v["component_mean_ms_runner_tm"].items()})
    for k, v in out["timing"]["batched"].items():
        print(k, {n: (round(v[n]["wall_s_median"], 3), round(v[n]["relative_change_pct"], 1), [round(x, 3) for x in v[n]["paired_diff_ci95_s"]]) for n in ("gated_fullbatch", "gated_selective")})


if __name__ == "__main__":
    main()
