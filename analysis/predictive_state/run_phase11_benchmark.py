"""EXP-0013 Phase 1.1 benchmark validation (numpy only, no training). Subcommands (each < 15 min; results -> results/predictive_state_phase1_1/):
  select_dilation OUT        choose the wall-dilation radius of ObsA/ObsB on the TRAIN layout range (the only tuned parameter)
  parse_fidelity  OUT        how well can walls / target / goal be recovered from ONE visible frame (TRAIN + DEV, never TEST)
  paired          OUT [n]    privileged references + observation-matched baselines on identical DEV episodes, 3 conditions
  horizon         OUT [n] [anchor|grid] [B,..] [planner,..]   planning-horizon diagnostic (privileged planners, stated simulation budgets)
  consistency     OUT        env transition vs oracle transition (la_move) on random actions across all splits' layouts
The TEST layout range (2000-2199) is never used here.
"""
import json
import sys
import time

import numpy as np

from . import controllers_la as C
from . import controllers_obs as O
from . import env as E
from . import env_la as L
from . import planning_diag as P
from .run_la_benchmark import wilson

CONDS = {"LA_CLEAR": E.Cond("LA_CLEAR", 0.0, (5, 15), 0.0, 2),          # defined here (Phase 1.1); env_la.LA_CONDS is unchanged
         "LA_NOM": L.LA_CONDS["LA_NOM"], "LA_OCC": L.LA_CONDS["LA_OCC"]}
EVAL_SEED = 5001                  # same seed as Phase 1 (DEV)
SEL_SEED = 31337                  # TRAIN, dilation selection


def make_env(split, cond, n, seed):
    return L.LAEnv(n, split, cond, seed, t_limit=80)


def summarize(r, n):
    ok = r["success"] > 0
    fin = r["finish"]
    k65 = int((ok & (fin <= L.LA_T_PICK)).sum())
    return {"success_at_budget": k65 / n, "ci95": wilson(k65, n), "success_at_80": float(ok.mean()),
            "median_finish_successful": float(np.median(fin[ok])) if ok.any() else None,
            "fail_kinds": {k: int((r["fail_kind"] == k).sum()) for k in set(r["fail_kind"].tolist()) if k},
            "effort": r["effort"], "per_episode_success_at_budget": (ok & (fin <= L.LA_T_PICK)).astype(int).tolist()}


def select_dilation(out, n=60):
    res = {"split": "train", "seed": SEL_SEED, "n": n, "cond": "LA_CLEAR", "rule": "pick the dilation radius with the higher mean ObsA/ObsB success within the budget; tie -> larger", "runs": {}}
    for dil in (0, 1):
        sc = []
        for cls in (O.ObsA, O.ObsB):
            env = make_env("train", CONDS["LA_CLEAR"], n, SEL_SEED)
            t0 = time.perf_counter()
            r = O.run_episodes_obs(env, cls(dil))
            s = summarize(r, n)
            sc.append(s["success_at_budget"])
            res["runs"][f"{cls.name}|dil{dil}"] = {k: s[k] for k in ("success_at_budget", "success_at_80", "fail_kinds")} | {"wall_s": time.perf_counter() - t0}
            print(cls.name, dil, s["success_at_budget"], s["fail_kinds"], flush=True)
        res["runs"][f"mean|dil{dil}"] = float(np.mean(sc))
    res["selected_dilation"] = 1 if res["runs"]["mean|dil1"] >= res["runs"]["mean|dil0"] else 0
    json.dump(res, open(out, "w"), indent=1)
    print("selected", res["selected_dilation"])


def true_wall_mask(walls_i):
    m = np.zeros((E.RES, E.RES), bool)
    for k in range(L.NWALL):
        w = walls_i[k]
        m |= (np.abs(E._X - w[0]) < w[2] / 2) & (np.abs(E._Y - w[1]) < w[3] / 2) & (w[2] > 0)
    return m


def parse_fidelity(out, n=150):
    res = {}
    for split, seed in (("train", 99), ("dev", 98)):
        env = make_env(split, CONDS["LA_CLEAR"], n, seed)
        obs = env.reset()
        iou, tgt_err, goal_err, tgt_vis, goal_vis, iou_filled = [], [], [], 0, 0, []
        for i in range(n):
            pr = O.parse(obs["img"][i], int(obs["task"][i].argmax()))
            tw = true_wall_mask(env.walls[i])
            iou.append(float((pr["wall"] & tw).sum() / max((pr["wall"] | tw).sum(), 1)))
            if pr["tgt"] is not None:
                tgt_vis += 1
                tgt_err.append(float(np.linalg.norm(pr["tgt"] - env.pos[i, 0])))
            if pr["goal_bbox"] is not None:
                goal_vis += 1
                goal_err.append(float(np.linalg.norm(O.bbox_center(pr["goal_bbox"]) - env.goal[i])))
        # shortest-path agreement: path length from spawn to goal on the parsed (dilated) map vs the true oracle field
        gap = []
        for i in range(min(n, 60)):
            pr = O.parse(obs["img"][i], int(obs["task"][i].argmax()))
            if pr["goal_bbox"] is None:
                continue
            gc = O.bbox_center(pr["goal_bbox"])
            f = O.geodesic(O.dilate(pr["wall"], 1), O.nearest_free(O.dilate(pr["wall"], 1), O.cell_of(gc)))
            v, _ = O.value(f, env.ee[i])
            fo, _, _ = L.geodesic_field(env.layout_ids[i], env.goal[i])
            vo = float(L.lookup(fo, env.ee[i]))
            gap.append(v - vo)
        res[split] = {"n": n, "wall_iou_mean": float(np.mean(iou)), "wall_iou_min": float(np.min(iou)),
                      "target_visible_frac": tgt_vis / n, "target_pos_err_mean": float(np.mean(tgt_err)), "target_pos_err_p95": float(np.percentile(tgt_err, 95)),
                      "goal_found_frac": goal_vis / n, "goal_center_err_mean": float(np.mean(goal_err)), "goal_center_err_p95": float(np.percentile(goal_err, 95)),
                      "path_length_error_parsed_minus_true_mean": float(np.mean(gap)), "path_length_error_abs_p95": float(np.percentile(np.abs(gap), 95))}
        print(split, res[split], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def paired(out, n=200, dil=None):
    if dil is None:
        dil = json.load(open(out.replace("paired", "dilation_selection")))["selected_dilation"]
    ctrls = [("greedy", "priv", lambda: C.Greedy()), ("reactive_memory", "priv", lambda: C.Reactive(True)), ("oracle_descent", "priv", lambda: C.Oracle("descent")),
             ("obs_A_single_frame", "obs", lambda: O.ObsA(dil)), ("obs_B_map_memory", "obs", lambda: O.ObsB(dil))]
    res = {"split": "dev", "seed": EVAL_SEED, "n": n, "budget": L.LA_T_PICK, "dilation": dil, "conds": {}}
    for cn, cond in CONDS.items():
        res["conds"][cn] = {}
        start = None
        for name, kind, mk in ctrls:
            env = make_env("dev", cond, n, EVAL_SEED)
            t0 = time.perf_counter()
            ctrl = mk()
            r = C.run_episodes(env, ctrl, T=82) if kind == "priv" else O.run_episodes_obs(env, ctrl)
            s = summarize(r, n)
            s["wall_s"] = time.perf_counter() - t0
            res["conds"][cn][name] = s
            if start is None:
                start = env.layout_ids.tolist()
            print(cn, name, s["success_at_budget"], s["success_at_80"], s["fail_kinds"], round(s["wall_s"], 1), flush=True)
        res["conds"][cn]["_layout_ids_first_controller"] = start
    json.dump(res, open(out, "w"), indent=1)


def _run_plan(res, out, pk, cost, agg, H, B, n, seed=None):
    env = make_env("dev", CONDS["LA_NOM"], n, EVAL_SEED)
    ctrl = P.PlanDiag(pk, H, B, cost=cost, agg=agg, seed=H if seed is None else seed)
    t0 = time.perf_counter()
    r = C.run_episodes(env, ctrl, T=82)
    s = summarize(r, n)
    s["wall_s"] = time.perf_counter() - t0
    s["sim_steps_per_decision"] = r["effort"]["sim_steps"] / max(r["effort"]["decisions"], 1)
    res["runs"][ctrl.name] = s
    print(ctrl.name, s["success_at_budget"], round(s["sim_steps_per_decision"]), round(s["wall_s"], 1), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def horizon(out, n=80, part="grid", budgets=(320,), planners=("rs_walk", "rs_const", "cem", "beam")):
    """part='anchor': rs_walk, geodesic, terminal, B = 64*H, n episodes (default 200 = the Phase-1 episode set): must reproduce Phase 1's
    oracle_shoot_geodesic numbers (proves PlanDiag == controllers_la.Oracle).
    part='grid'  : planner x cost x H at the stated simulation budgets (sim steps per decision)."""
    res = {"split": "dev", "seed": EVAL_SEED, "n": n, "cond": "LA_NOM", "budget_steps": L.LA_T_PICK, "part": part, "runs": {}}
    if part == "anchor":
        for H in (1, 5, 10, 20):
            _run_plan(res, out, "rs_walk", "geodesic", "terminal", H, 64 * H, n)
        return
    for B in budgets:
        for pk in planners:
            for cost in ("euclid", "geodesic"):                       # euclid first: it is the only arm that can show a lookahead benefit
                for agg in (("terminal", "sum") if pk == "rs_walk" else ("terminal",)):
                    for H in (1, 5, 10, 20):
                        _run_plan(res, out, pk, cost, agg, H, B, n)


def consistency(out, steps=60):
    res = {}
    rng = np.random.default_rng(0)
    for split in ("train", "dev"):                  # the TEST range is never touched
        env = L.LAEnv(64, split, CONDS["LA_NOM"], 17)
        pred = env.ee.copy()
        worst = 0.0
        for _ in range(steps):
            a = np.concatenate([rng.uniform(-1.4, 1.4, (64, 2)), np.zeros((64, 1))], 1).astype(np.float32)
            live = ~env.done
            nxt = L.la_move(pred, L.cap_step(a[:, :2]), env.walls)
            env.step(a)
            nxt = np.where(live[:, None], nxt, pred)
            worst = max(worst, float(np.abs(nxt - env.ee).max()))
            pred = nxt
        res[split] = {"layouts": 64, "steps": steps, "max_abs_position_error": worst}
        print(split, res[split], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    a = sys.argv[3:]
    if mode == "horizon":     # horizon OUT [n] [anchor|grid] [B,B] [planner,planner]
        horizon(out, int(a[0]) if a else 80, a[1] if len(a) > 1 else "grid", tuple(int(x) for x in a[2].split(",")) if len(a) > 2 else (320,),
                tuple(a[3].split(",")) if len(a) > 3 else ("rs_walk", "rs_const", "cem", "beam"))
    else:
        {"select_dilation": select_dilation, "parse_fidelity": parse_fidelity, "paired": paired, "consistency": consistency}[mode](out, *[int(x) for x in a])
