"""EXP-0013 Phase 1.1 closeout analysis. Derives EVERY number in the closeout tables from committed raw files:
  results/predictive_state_phase1_1/paired.json                          (historical, shared-RNG v1; read only)
  results/predictive_state_phase1_1_closeout/repro_v1_<COND>.json         (v1 reproduced with unchanged code + passive valid log)
  results/predictive_state_phase1_1_closeout/paired_replay_<COND>.json    (paired-replay protocol)
Usage: python -m analysis.predictive_state.analyze_closeout [RESULTS_DIR]   -> closeout_summary.json + markdown on stdout
STATS_VERSION documents the method: exact McNemar (reported as '0 discordant pairs' when there are none), Wilson episode-level interval,
layout-clustered percentile bootstrap (resample the unique layouts, B=4000, seed 0, ratio estimator), and, when the observed value has no
variation, NO inferential bound (closeout-stats-2): the observation is reported descriptively as "k/n successes across m unique DEV layouts".
closeout-stats-1 also printed an exact Clopper-Pearson "failure <= 0.034" over the 87 unique layouts. That bound assumes independent layouts drawn
from the population of interest; the DEV pool has only 100 ids, 87 of them sampled with replacement, several spawns share a layout, and the
claim would concern unseen layouts. It was withdrawn; the superseded outputs are kept in superseded_stats1/."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from .run_la_benchmark import wilson

STATS_VERSION = "closeout-stats-2"
CONDS = ("LA_CLEAR", "LA_NOM", "LA_OCC")
CTRLS = ("greedy", "reactive_memory", "oracle_descent", "obs_A_single_frame", "obs_B_map_memory")
PAIRS = (("obs_A_single_frame", "obs_B_map_memory"), ("obs_B_map_memory", "oracle_descent"), ("obs_A_single_frame", "oracle_descent"),
         ("obs_B_map_memory", "reactive_memory"), ("obs_A_single_frame", "greedy"))
BUDGET = 65
B = 4000


def mcnemar(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_exact": None, "note": "0 discordant pairs: no test is defined"}
    k = min(n10, n01)
    return {"a_only": n10, "b_only": n01, "discordant": n, "p_exact": min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), "note": ""}


def cluster_boot(layouts, x, y=None, seed=0):
    """Layout-clustered percentile bootstrap of mean(x) (or mean(x)-mean(y)); returns (lo, hi) or None if the statistic never varies."""
    layouts, x = np.asarray(layouts), np.asarray(x, float)
    y = None if y is None else np.asarray(y, float)
    ul = np.unique(layouts)
    idx = {l: np.nonzero(layouts == l)[0] for l in ul}
    sx = np.array([x[idx[l]].sum() for l in ul]); sy = None if y is None else np.array([y[idx[l]].sum() for l in ul])
    cnt = np.array([len(idx[l]) for l in ul], float)
    g = np.random.default_rng(seed)
    draws = g.integers(0, len(ul), (B, len(ul)))
    den = cnt[draws].sum(1)
    est = sx[draws].sum(1) / den
    if y is not None:
        est = est - sy[draws].sum(1) / den
    if np.ptp(est) == 0:
        return None
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def finish_stats(fin, ok):
    f = np.asarray(fin)[np.asarray(ok) > 0]
    if len(f) == 0:
        return None
    return {"n_success": int(len(f)), "min": int(f.min()), "p25": float(np.percentile(f, 25)), "median": float(np.median(f)), "p75": float(np.percentile(f, 75)),
            "p95": float(np.percentile(f, 95)), "max": int(f.max()), "n_over_65": int((f > BUDGET).sum())}


def first_diff_counts(logs_a, logs_b):
    """Episodes whose valid sequences differ anywhere on the common prefix of two runs (includes steps after an episode ended)."""
    L = min(len(logs_a[0]), len(logs_b[0]))
    return int(sum(a[:L] != b[:L] for a, b in zip(logs_a, logs_b))), L


def live_diff_counts(logs_a, logs_b, fin_a, fin_b):
    """Episodes whose blackout sequence differed over the observations BOTH controllers acted on: indices 0 .. min(finish_a, finish_b) - 1
    (index 0 = reset observation). Draws after an episode ended cannot change any outcome, so this is the count that matters for pairing."""
    n = 0
    for a, b, fa, fb in zip(logs_a, logs_b, fin_a, fin_b):
        w = min(fa if fa > 0 else len(a), fb if fb > 0 else len(b))
        n += a[:w] != b[:w]
    return int(n)


def reproduction(hist, repro):
    out = {}
    for cn in CONDS:
        rows = {}
        for c in CTRLS:
            h, r = hist["conds"][cn][c]["per_episode_success_at_budget"], repro[cn]["controllers"][c]["success_at_budget"]
            rows[c] = {"identical_per_episode_success_at_65": h == r, "n_differing_episodes": int(sum(x != y for x, y in zip(h, r))),
                       "layout_ids_identical": hist["conds"][cn]["_layout_ids_first_controller"] == repro[cn]["controllers"][c]["layout_ids"]}
        out[cn] = rows
    return out


def v1_defect(repro):
    """How many episodes saw a different blackout schedule under different controllers in the historical (shared RNG) protocol."""
    out = {}
    for cn in CONDS:
        cc = repro[cn]["controllers"]
        out[cn] = {}
        for a, b in PAIRS + (("greedy", "oracle_descent"),):
            n_diff, L = first_diff_counts(cc[a]["valid_log"], cc[b]["valid_log"])
            out[cn][f"{a}|{b}"] = {"episodes_with_different_valid_sequence_while_live": live_diff_counts(cc[a]["valid_log"], cc[b]["valid_log"], cc[a]["finish"], cc[b]["finish"]),
                                   "episodes_with_different_valid_sequence_whole_run": n_diff, "common_steps_compared_whole_run": L, "of": len(cc[a]["valid_log"])}
    return out


def paired_replay(pr, repro_v1, hist):
    out = {"conds": {}}
    for cn in CONDS:
        cc = pr[cn]["controllers"]
        ref = cc["greedy"]
        layouts = ref["layout_id"]
        spawn_same = all(cc[c]["spawn_digest"] == ref["spawn_digest"] for c in CTRLS)
        sched_same = all(cc[c]["schedule_digest"] == ref["schedule_digest"] for c in CTRLS)
        ids_same = all(cc[c]["episode_id"] == ref["episode_id"] and cc[c]["layout_id"] == layouts for c in CTRLS)
        L = min(cc[c]["n_observations"] for c in CTRLS)
        realized_same = all(all(a[:L] == b[:L] for a, b in zip(cc[c]["valid_log"], ref["valid_log"])) for c in CTRLS)
        sid = hashlib.sha1("".join(ref["schedule_digest"]).encode()).hexdigest()[:16]
        block = {"n_episodes": len(layouts), "unique_layouts": len(set(layouts)), "schedule_identifier": sid,
                 "all_controllers_same_episode_ids_layouts": ids_same, "all_controllers_same_spawn_digests": spawn_same,
                 "all_controllers_same_schedule_digests": sched_same, "realized_valid_identical_across_controllers_on_common_steps": realized_same,
                 "common_steps_compared": L, "realized_valid_equals_closed_form_schedule": {c: cc[c]["realized_valid_equals_schedule"] for c in CTRLS},
                 "blackout_schedule_first_65_steps": {
                     "dark_step_fraction": float(np.mean([[ch == "0" for ch in lg[1:66]] for lg in ref["valid_log"]])),
                     "episodes_with_any_blackout": float(np.mean(["0" in lg[1:66] for lg in ref["valid_log"]]))},
                 "controllers": {}, "pairs": {}}
        s65 = {c: np.array(cc[c]["success_at_budget"]) for c in CTRLS}
        for c in CTRLS:
            k, n = int(s65[c].sum()), len(layouts)
            v1_rate = float(np.mean(hist["conds"][cn][c]["per_episode_success_at_budget"]))
            fails = {kk: int(sum(1 for f in cc[c]["fail_kind"] if f == kk)) for kk in set(cc[c]["fail_kind"]) if kk}
            ci = cluster_boot(layouts, s65[c])
            block["controllers"][c] = {
                "success_at_65": k / n, "successes_at_65": k, "success_at_80": float(np.mean(cc[c]["success_at_80"])),
                "wilson95_episode_level": wilson(k, n),
                "layout_cluster_bootstrap95": ci if ci else None,
                "descriptive": f"{k}/{n} successes at 65 steps across {len(set(layouts))} unique DEV layouts (no claim for unseen layouts)",
                "finish": finish_stats(cc[c]["finish"], cc[c]["success_at_80"]), "fail_kinds_at_cap_80_or_before": fails,
                "effort": cc[c]["effort"], "v1_historical_success_at_65": v1_rate, "delta_vs_v1_historical": k / n - v1_rate,
                "per_episode_success_at_65_sha1": hashlib.sha1(bytes(s65[c].astype(np.uint8).tolist())).hexdigest()[:12]}
        for a, b in PAIRS:
            d = s65[a].mean() - s65[b].mean()
            block["pairs"][f"{a}|{b}"] = {"difference_a_minus_b": float(d), "layout_cluster_bootstrap95_difference": cluster_boot(layouts, s65[a], s65[b]),
                                          "mcnemar_episode_level": mcnemar(s65[a], s65[b])}
        out["conds"][cn] = block
    return out


def horizon_effort():
    """Re-tabulate the Phase 1.1 horizon files with the search structure made explicit (documentation-only; no new runs).
    planner|cost|agg|H|B -> success@65, simulated transitions per decision, and the candidate structure implied by the planner code
    (rs: N=B//H sequences of length H; cem: N=max(B//(H*4),2) per iteration x 4 iterations; beam: width W=max(B//(17*H),1), 17 expansions per node)."""
    rows = []
    for f, tag in (("horizon_anchor", "anchor(B=64*H, n=200)"), ("horizon_grid_B320", "grid B=320 (n=80)"), ("horizon_grid_B2720_euclid", "grid B=2720 euclid (n=80)")):
        h = json.load(open(f"results/predictive_state_phase1_1/{f}.json"))
        for name, s in h["runs"].items():
            pk, cost, agg, Hs, Bs = name.split("|")
            H, Bv = int(Hs[1:]), int(Bs[1:])
            cand = {"rs_walk": f"{max(Bv // H, 1)} sequences", "rs_const": f"{max(Bv // H, 1)} sequences", "cem": f"{max(Bv // (H * 4), 2)} x 4 iterations",
                    "beam": f"width {max(Bv // (17 * H), 1)} x 17 expansions"}[pk]
            rows.append({"set": tag, "planner": pk, "cost": cost, "agg": agg, "H": H, "B": Bv, "n": h["n"], "success_at_65": s["success_at_budget"],
                         "sim_steps_per_decision": s["sim_steps_per_decision"], "candidates": cand})
    return rows


def fmt_ci(ci):
    return "n/a (no variation)" if ci is None else "[%.3f, %.3f]" % tuple(ci)


def markdown(out):
    L = ["# Closeout tables (generated by analysis/predictive_state/analyze_closeout.py from the raw files; do not edit)", ""]
    L += ["## Reproduction of the historical per-episode outcomes with unchanged code", "", "| cond | controllers with identical per-episode success@65 (of 5) |", "|---|---|"]
    for cn, rows in out["reproduction_of_historical"].items():
        L.append(f"| {cn} | {sum(r['identical_per_episode_success_at_65'] for r in rows.values())} |")
    L += ["", "## Historical (v1 shared RNG) defect: episodes (of 200) whose blackout sequence differed between two controllers", "",
          "Headline = while live under both controllers (observations both acted on); in brackets = over the whole common run length (includes steps after an episode ended).", "", "| pair | LA_CLEAR | LA_NOM | LA_OCC |", "|---|---|---|---|"]
    for k in out["historical_defect_measured"]["LA_NOM"]:
        kk = k.replace(chr(124), " vs ")
        L.append(f"| {kk} | " + " | ".join("%d (%d)" % (out["historical_defect_measured"][cn][k]["episodes_with_different_valid_sequence_while_live"], out["historical_defect_measured"][cn][k]["episodes_with_different_valid_sequence_whole_run"]) for cn in CONDS) + " |")
    for cn, b in out["paired_replay"]["conds"].items():
        L += ["", f"## Paired replay, {cn}: {b['n_episodes']} episodes, {b['unique_layouts']} unique layouts, schedule id `{b['schedule_identifier']}`, dark steps {b['blackout_schedule_first_65_steps']['dark_step_fraction']:.3f}, episodes with a blackout {b['blackout_schedule_first_65_steps']['episodes_with_any_blackout']:.3f}", "",
              "| controller | success@65 | success@80 | Wilson95 (episodes) | layout-cluster bootstrap95 | descriptive (successes / episodes; unique layouts) | median finish (successes) | timeouts | v1 hist @65 | delta |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c, v in b["controllers"].items():
            ub = f"{v['successes_at_65']}/{b['n_episodes']}; {b['unique_layouts']} layouts"
            L.append(f"| {c} | {v['success_at_65']:.3f} | {v['success_at_80']:.3f} | [{v['wilson95_episode_level'][0]:.3f}, {v['wilson95_episode_level'][1]:.3f}] | {fmt_ci(v['layout_cluster_bootstrap95'])} | {ub} | "
                     f"{v['finish']['median']} | {v['fail_kinds_at_cap_80_or_before'].get('timeout', 0)} | {v['v1_historical_success_at_65']:.3f} | {v['delta_vs_v1_historical']:+.3f} |")
        L += ["", "| pair (a vs b) | a-b success@65 | layout-cluster bootstrap95 of a-b | a-only / b-only discordant | exact McNemar p (episode level) |", "|---|---|---|---|---|"]
        for k, v in b["pairs"].items():
            m = v["mcnemar_episode_level"]
            L.append(f"| {k.replace('|', ' vs ')} | {v['difference_a_minus_b']:+.3f} | {fmt_ci(v['layout_cluster_bootstrap95_difference'])} | {m['a_only']} / {m['b_only']} | "
                     f"{('%.3g' % m['p_exact']) if m['p_exact'] is not None else m['note']} |")
    L += ["", "## Horizon diagnostic re-tabulated (Phase 1.1 files; no new runs)", "", "| set | planner | cost | agg | H | B | candidates | sim steps/decision | success@65 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out["horizon_effort"]:
        L.append(f"| {r['set']} | {r['planner']} | {r['cost']} | {r['agg']} | {r['H']} | {r['B']} | {r['candidates']} | {r['sim_steps_per_decision']:.0f} | {r['success_at_65']:.3f} |")
    return chr(10).join(L) + chr(10)


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/predictive_state_phase1_1_closeout")
    hist = json.load(open("results/predictive_state_phase1_1/paired.json"))
    repro = {cn: json.load(open(d / f"repro_v1_{cn}.json")) for cn in CONDS}
    pr = {cn: json.load(open(d / f"paired_replay_{cn}.json")) for cn in CONDS}
    out = {"stats_version": STATS_VERSION, "bootstrap_B": B, "reproduction_of_historical": reproduction(hist, repro), "historical_defect_measured": v1_defect(repro),
           "paired_replay": paired_replay(pr, repro, hist), "horizon_effort": horizon_effort()}
    json.dump(out, open(d / "closeout_summary.json", "w"), indent=1)
    (d / "closeout_tables.md").write_text(markdown(out), encoding="utf-8")
    print("== reproduction of the historical per-episode outcomes (unchanged code)")
    for cn, rows in out["reproduction_of_historical"].items():
        print(cn, {c: (r["identical_per_episode_success_at_65"], r["n_differing_episodes"]) for c, r in rows.items()})
    print("== historical defect: episodes with different blackout sequences between controllers")
    for cn, rows in out["historical_defect_measured"].items():
        print(cn, {k: (v["episodes_with_different_valid_sequence_while_live"], v["episodes_with_different_valid_sequence_whole_run"]) for k, v in rows.items()})
    print("== paired replay")
    for cn, blk in out["paired_replay"]["conds"].items():
        print(cn, "layouts", blk["unique_layouts"], "sched", blk["schedule_identifier"], "invariants",
              blk["all_controllers_same_spawn_digests"], blk["all_controllers_same_schedule_digests"], blk["realized_valid_identical_across_controllers_on_common_steps"])
        for c, v in blk["controllers"].items():
            print("  ", c, v["success_at_65"], v["success_at_80"], "wilson", [round(x, 3) for x in v["wilson95_episode_level"]], "cluster", v["layout_cluster_bootstrap95"],
                  "desc", v["descriptive"], "delta_v1", round(v["delta_vs_v1_historical"], 3))
        for k, v in blk["pairs"].items():
            print("  ", k, round(v["difference_a_minus_b"], 3), v["layout_cluster_bootstrap95_difference"], v["mcnemar_episode_level"])


if __name__ == "__main__":
    main()
