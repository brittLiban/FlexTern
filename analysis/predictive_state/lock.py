"""Write protocol_v1.json: constants, gates, and code hashes, BEFORE any learned-policy measurement (EXP-0012).

Usage: python -m analysis.predictive_state.lock   (from the repo root). Re-running after the lock commit must reproduce
the same hashes unless an amendment was made (amendments are appended in the experiment record, never silent).
"""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["env.py", "expert.py", "quant.py", "models.py", "train.py", "evaluate.py", "cost_model.py"]

PROTOCOL = {
    "protocol_version": 1,
    "experiment": "EXP-0012",
    "benchmark": {"name": "OccPick2D", "res": 32, "max_step": 0.05, "pick_r": 0.06, "goal_half": 0.10, "t_pick": 80,
                  "layout_ranges": {"train": [0, 400], "dev": [1000, 1100], "test": [2000, 2200]},
                  "test_conditions": {"NOM": "p_blackout .02 dur 5-15, no displacement, K=2",
                                      "OCC": "p_blackout .05 dur 10-30, no displacement, K=3",
                                      "DIST": "p_blackout .02 dur 5-15, displacement always, K=2",
                                      "HARD": "p_blackout .05 dur 10-30, displacement always, K=4 (unseen in training)",
                                      "LONG": "5 sequential picks, p_blackout .03 dur 5-20, displacement p=.5 per pick, K=2, no state reset"}},
    "training": {"method": "behaviour cloning from a noisy privileged scripted expert, identical data/steps/batch/lr for every arm",
                 "train_episodes": 2000, "train_seed": 0, "T": 160, "steps": 3000, "batch": 24, "lr": 2e-3,
                 "optimizer": "AdamW wd 1e-4, OneCycle (10% warmup)", "obs_dropout": "burst p=.05 len 1-8, all arms",
                 "jepa": "EMA(0.995) target encoder, stop-grad, cosine loss, weight 1.0, one-step predictor on h_t and a_t",
                 "seeds": [0, 1, 2]},
    "arms": {"R": "single frame", "S4": "4-frame stack", "T16": "causal transformer, 16-step window",
             "G0": "GRU, no JEPA", "G": "GRU + JEPA, FP16-costed (= Baselines B and D)",
             "G-I8a": "INT8 W/A QAT, encoder+updater", "G-I4u": "INT4 W QAT updater only", "G-I4a": "INT4 W QAT all",
             "G-Tu": "ternary W QAT updater only (Candidate E1)", "G-Ta": "ternary W QAT encoder+updater (E2)",
             "G-Ta-s8": "G-Ta with INT8 recurrent state (state-precision ablation)"},
    "eval": {"seeds_per_arm": 3, "episodes_per_condition_per_seed": {"NOM": 300, "OCC": 300, "DIST": 300, "HARD": 300, "LONG": 100},
             "split_use": "dev for all design/debug/calibration; test touched once, after this lock and after dev-based "
                          "threshold calibration, for the reported numbers",
             "macro_success": "mean over NOM, OCC, DIST, HARD",
             "ci": "bootstrap over (seed, episode) pooled, 2000 draws, plus per-seed values reported"},
    "adaptive": {"triggers": ["full", "fixed:k", "pix:tau (8x8 pooled pixel RMS change vs last encoded frame)",
                              "learned:tau (surprise head on h, a, 8x8 pooled frame)"],
                 "shared_forced_encode": "first frame, task change, slip (held lost while gripper closed), >=8 steps since last encode",
                 "budgets": [0.5, 0.33, 0.2], "calibration": "tau chosen on DEV (NOM+DIST mix) to hit each target encoder fraction",
                 "info_rule": "gate may use only proprioception, the task code, the recurrent state and 8x8 pooled raw pixels; "
                              "never the encoder embedding of the current frame, never privileged state"},
    "cost_metrics": {"analytical": "MACs and weight+state bytes per control step, analysis/predictive_state/cost_model.py; "
                                   "ternary MAC counted as an add",
                     "measured": "batch-1 CPU per-step latency (p50/p95/max) of the PyTorch reference implementation, laptop on AC power; "
                                 "ternary has no packed kernel here, so measured latency cannot credit 2-bit weights"},
    "gates": {
        "G0_discrimination": "On DEV: expert success >= 0.95 in every condition; and mean(G success on OCC,HARD) - max(R,S4 success on OCC,HARD) >= 0.15. "
                             "Else STOP (benchmark does not establish a meaningful experiment).",
        "G1_floor": "G (FP16 recurrent) test macro success >= 0.60 and NOM success >= 0.70 (mean over seeds).",
        "G2_ternary_retention": "G-Tu and G-Ta each: test macro success >= 0.95 x G macro success (mean over seeds) with the lower 95% bootstrap bound of the ratio >= 0.90, and absolute macro >= 0.60.",
        "G3_adaptive": "Learned-trigger model at the 0.33 budget (>=2x fewer encoder runs than full perception): macro success >= 0.95 x its own full-perception success, DIST success >= 0.90 x its full-perception DIST success, and success at matched budget >= both fixed-rate and pixel-difference controls by >= 0.03 (else the learned trigger is recorded as giving no advantage).",
        "G4_cost": "Analytical (labelled ANALYTICAL): F-ternary vs the strongest suitable control = INT4-QAT recurrent (G-I4a or G-I4u, whichever is cheaper while retaining >=95% of G macro success) under the SAME trigger and budget: >=2x lower bytes per step AND >=2x lower MACs per step. Also reported vs FP16 full perception. Measured latency ratio reported separately; 'MEASURED PASS' requires measured p50 ratio >= 2x.",
        "G5_stability": "LONG (5x training horizon, no state reset): G-T* picks_done fraction >= 0.95 x G; linear-probe R^2 of occluded target position (probe fit on steps <= 100 of dev rollouts, scored on steps > 300 of test rollouts) drops <= 0.05 vs G; no NaN and max|h| <= 1.0 by construction (tanh GRU) so also report mean |h| drift.",
        "G6_vs_best_conventional": "Any advantage claim must hold against the best of {G-I8a, G-I4u, G-I4a} that retains >=95% of G macro success.",
    },
    "verdict_rules": {
        "STOP": "G0 fails, or G1 fails after one documented fix attempt, or persistent state gives no benefit over R/S4 AND neither adaptivity nor ternary gates pass.",
        "CONTINUE": "G1, G2, G3, G5 pass and G4 passes (analytical).",
        "NARROW": "G1 passes and G2, G5 pass but only one of G3/G4 passes (name which).",
        "CHANGE_ARCHITECTURE": "G1 passes and persistent state helps, but G2 or G5 fails (ternary unstable/lossy), or G4 fails against the INT4 control.",
    },
    "questions": {
        "A_persistent_state": "G vs R/S4/T16 on OCC/HARD/DIST, and encoder reduction under G3.",
        "B_jepa": "G vs G0 paired by seed: macro success and probe R^2; collapse monitored by embedding std.",
        "C_ternary": "G vs G-I8a/G-I4*/G-T*; long-horizon stability (G5); state precision (G-Ta-s8).",
        "D_adaptive": "G3 with triggers fixed-rate, pixel-diff, learned, for FP16 and ternary and INT4 models.",
    },
    "not_authorized": "silicon, FPGA, RTL redesign, hardware purchases, paid cloud, physical robots, energy measurement",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rec = dict(PROTOCOL)
    rec["code_sha256"] = {f: sha(HERE / f) for f in FILES}
    try:
        rec["git_head_at_lock"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        rec["git_head_at_lock"] = "unknown"
    rec["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import numpy
        import torch
        rec["environment"].update({"numpy": numpy.__version__, "torch": torch.__version__,
                                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception:
        pass
    out = HERE / "protocol_v1.json"
    out.write_text(json.dumps(rec, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
