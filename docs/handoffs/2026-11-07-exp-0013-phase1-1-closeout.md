# Handoff

Date: 2026-11-07
Agent: Claude (implementation agent)
Current commit: see `git log exp-0013-phase1-1-closeout` (branch from `main` @ 6cbd44b; **not merged**)

## Objective

EXP-0013 Phase 1.1 closeout: repair the paired-evaluation defect, correct the statistics and reporting, re-verify inference and latency. No Phase 1.2, no training.

## What I Completed

Pairing defect reproduced and measured with unchanged code, paired-replay environment with 10 tests, corrected diagnostic re-evaluation (obs_B 200/200 in LA_CLEAR/NOM/OCC), layout-clustered statistics, TEST-exposure audit and guard, inference re-verification (34 tests), AC-power latency repeat, tiered test runner, governance and status corrections. Verdict and gates: `docs/experiments/EXP-0013-phase1-1-closeout.md`.

## What I Changed

New: `analysis/predictive_state/{env_paired,closeout_repro_v1,closeout_paired,analyze_closeout,closeout_manifest}.py`, `split_manifest.json`, `tests/{test_env_paired,test_split_guard}.py`, `scripts/{run_tests.py,capture_power_state.ps1}`, `tests/test_budget.json`, `results/predictive_state_phase1_1_closeout/`. Edited: `phase11_timing.py` (trace split defaults to TRAIN), `AGENTS.md` §10, `tests/README.md`, `docs/{CURRENT_WORK,PROJECT_STATE,KNOWN_ISSUES}.md`, correction notes on the Phase 1 and 1.1 records. Not changed: `env.py`, `env_la.py`, controllers, `infer.py`, `evaluate.py`, `models.py`, `train.py`, historical results.

## Tests / Validation Run

See the closeout record section 11 and `results/predictive_state_phase1_1_closeout/test_logs/` (observed this session, not carried over).

## Actual Results

MEASURED: v1 reproduces exactly; 114-130/200 (LA_NOM) and 177-192/200 (LA_OCC) episodes had different blackout sequences between controllers while both were acting under v1; paired replay: obs_B 1.000/1.000/1.000, obs_A 0.995/0.970/0.560, oracle 1.000 (87 unique layouts). Latency on host-reported AC: lean legacy skipping -28% batch-1, shared runner -4%, batched CPU -9..-25%, batched CUDA +6..+16%; energy not measured.

## Remaining Problems

Section 13 of the closeout record. Founder decisions pending: merge of this branch; authorization of Phase 1.2 (v2-B with v2-A control) and a fresh held-out layout range.

## Important Context

- Use `PairedLAEnv` (never `LAEnv`) for any paired comparison; do not use ids 2000-2199 for v2 claims; the `"test"` literal is guarded by `tests/test_split_guard.py`.
- Windows host Python 3.12 imports torch 2.11 now (CUDA works); WSL was used for the timing run only (matching the historical setup).
- Spawns in the paired protocol are not bit-identical to v1; compare aggregates, not episodes, across protocols.

## Addendum: hardening round after the independent code audit

See `docs/experiments/EXP-0013-phase1-1-closeout.md` section 15. Net effect for a continuing agent: `PairedLAEnv` custom `episode_ids` need explicit `layout_ids`; use `analysis/predictive_state/split_policy.py` for any split choice (TRAIN default); do not cite a layout-level failure bound (only "k/n across m unique DEV layouts"); `RecurrentRunner` conflates sensor blackout with chosen skip and does not re-arm the first observation (KI-031), so a v2 interface needs an explicit observation status.
