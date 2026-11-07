# EXP-0013 Phase 1 — Inference repair, measurement integrity, lookahead benchmark

> **Phase 1.1 addendum (2026-11-07):** the data below are unchanged, but some interpretations are superseded - Gate C is downgraded (the benchmark is solved by a non-learned observation-matched baseline), the `rec_updates` counter and the latency reading are corrected, and the shooting degradation is attributed to the sampler/objective. See `EXP-0013-phase1-1.md`.

Date: 2026-11-07 · Branch `exp-0013-predictive-planning` · Authorization: Phase 1 only. No learned world model, no training, no quantization, no merge. Audit: `EXP-0013-audit.md`. Labels per AGENTS.md §7.

## 1. What changed

| Area | Change |
|---|---|
| `analysis/predictive_state/infer.py` (new) | `RecurrentRunner`: one stateful, instrumented step (trigger decision → gather/scatter encoder → recurrent update → policy). `gated_encode` encodes only eligible rows and never calls the encoder for an empty set. `Counters`: encoder calls vs samples, recurrent updates, active steps, blackout / trigger-skipped / finished rows, per-component time. |
| `evaluate.py` | `rollout` (recurrent arms) now uses the runner; the unmodified EXP-0012 function is kept as `rollout_legacy` (reference). Non-recurrent arms still use the legacy path. |
| `latency.py` | Rebuilt on the same runner (batch 1): `fixed:K` now works, blackout frames are never encoded, `full` no longer pays for pooling it does not use, per-component timing and counters are reported. |
| `dagger.py` | Encoder gated by validity/alive (outputs unchanged; pinned by a CPU equivalence test). Collection was **not** re-run. |
| `models.py` / `train.py` | **Untouched** (training masking, EMA target and model interface unchanged). |
| Observation status | Blackout, trigger-skipped and finished are distinguished **in counters only**; the network still gets one `valid` flag (decision D2 not taken). |
| `env_la.py`, `controllers_la.py`, `run_la_benchmark.py` (new) | OccPick2D-LA v1 and its privileged controls (numpy only; `env.py` untouched). |
| Tests (new) | `tests/test_inference_skip.py` (24, need torch/WSL), `tests/test_env_la.py` (16, numpy). |

Note: `lock.py` hashed `evaluate.py` for the EXP-0012 protocol; that locked version is in git history (commit f75b95a and ancestors). EXP-0012 results were produced by `rollout_legacy` semantics and are not reinterpreted.

## 2. Gate A/B evidence — encoder skipping (MEASURED)

G_s0 (FP16 recurrent, EXP-0012 checkpoint), 32 DEV episodes, condition NOM, seed 5, GPU; counts from a spy on `net.enc.forward` (`results/predictive_state_phase1/before_after_encoder_work.json`, script `phase1_measure.py`):

| trigger | legacy: encoder calls / **images executed** | gated: calls / **images executed** | logical requests (both) | per-episode success identical |
|---|---|---|---|---|
| full | 80 / 2560 | 80 / **1303** | 1303 | yes (0.656) |
| fixed:0.25 | 80 / 2560 | 49 / **331** | 331 | yes (0.719) |
| pix:0.03 | 80 / 2560 | 72 / **313** | 313 | yes (0.719) |

Executed work now equals the logical request count exactly. Calls are not samples: 49 calls moved 331 images. Wall time of the batched GPU evaluation did **not** improve (0.65–0.84 s for both; the batch is small and launch-bound, gather/scatter adds ops). Skipping reduces *executed work*, not wall time, in this setting.

Batch-1 CPU latency (`latency_before_after.json`, AC power confirmed, 10 DIST episodes, noisy at sub-millisecond scale): the EXP-0012 batch-1 path already skipped, so this is **not** a measurement of the fix. Decisions are identical (same `enc_frac` in every row). The rebuilt path is ~0.1–0.3 ms/step slower in skip modes (e.g. learned@0.5 p50 0.42 → 0.71 ms, pix@0.33 0.40 → 0.52 ms; `full` 0.63 → 0.62 ms) because of counter/gating bookkeeping on a ~0.5 ms step. Do not read this as a speed claim in either direction; the timing path should be re-benchmarked with counters disabled before any latency claim (KI-025).

Tests (all passing): all encoded; none encoded (no call, state still advances); mixed batch (encoder input equals exactly the eligible rows); blackout rows never encoded even when forced; finished rows never encoded; fixed-rate schedule exact (encodes at t=0,3,7,11,15); first frame, max-skip (t=0,9,18), task-change and slip forced observations; learned trigger threshold on a stubbed head; pixel-change trigger; batched runner equals per-row n=1 runner (decisions and states); full-perception equals the original computation step by step; new `rollout` equals `rollout_legacy` (decisions, success, alive steps) for full/fixed/pix; true samples equal the legacy logical count; determinism; latency path uses the same runner; DAgger gated = encode-everything.

Causality preserved: `a_prev` is the previous executed action; the state advances every step from the zero embedding plus `a_prev` and proprioception when vision is skipped.

## 3. OccPick2D-LA v1 (benchmark)

Task: EXP-0012 pick-and-place, plus a vertical divider (x≈0.40, 0.08 thick ≥ 2 px, distinct colour, drawn over occluders) with **two gaps**: an open gap and a decoy gap that opens into a closed pocket (0.20 deep). Agent, target and distractors start left; the goal zone is right, so every success crosses the divider. The end-effector slides along walls (per-axis blocking). Actions: Δx,Δy∈[-1,1]×0.05, grip. Success: release in the goal zone within the budget; failures: wrong pick, timeout. Occlusion/blackouts of EXP-0012 are retained (conditions LA_NOM, LA_OCC); displacement events are not supported in v1 (a teleported target could land in a wall). Single dynamics function `la_move` is shared by env and oracle planners. Step budget **LA_T_PICK = 65**, locked as ceil(1.5 × p95 oracle finish steps) on the TRAIN layout range *before any baseline was evaluated on DEV* (`la_calibration.json`: oracle p50 31, p95 43, max 54). Caveat: I did see baseline outcomes on TRAIN layouts while debugging the controllers; the budget rule used only oracle data. Layout ids use the EXP-0012 disjoint ranges (train 0–399, dev 1000–1099, test 2000–2199). **The TEST range was not used.**

### Information available to each controller

| Variable | Environment | Oracle (planners) | Reactive / greedy baselines | Future learned controller |
|---|---|---|---|---|
| wall map / pocket | yes | yes (geodesic field) | **no** — rays ≤ 0.25 only | no (must see them in the image; they are rendered and stay visible except in blackouts) |
| true ee, target, goal, held | yes | yes | yes (privileged) | no: proprioception (x,y,held,grip) + task colour + 32×32 image only; target/goal must be inferred from pixels/memory |
| true dynamics (`la_move`) | yes | yes | no | no (learned) |
| expert gripper rule | – | yes | yes | no (learned) |

### Controllers and results (DEV, 200 episodes, seed 5001, same layouts/spawns for every controller; `la_dev_results.json`)

| controller | success within budget 65 (95% Wilson) | success within 80 | median finish (successes) | sim steps / decision | Dijkstra node expansions (whole run) |
|---|---|---|---|---|---|
| greedy (slide) | 0.240 [0.19, 0.30] | 0.240 | 29.5 | 0 | 0 |
| reactive wall-follower (local rays) | 0.260 [0.20, 0.32] | 0.260 | 29.0 | 0 | 0 |
| reactive + retreat memory | 0.295 [0.24, 0.36] | 0.435 | 41.0 | 0 | 0 |
| **oracle shortest-path descent** (H=1, map value) | **1.000** [0.98, 1.00] | 1.000 | 31.0 | 18 | 1.36 M |
| oracle shooting, Euclid cost, H=1 / 5 / 10 / 20 | 0.240 / 0.205 / 0.000 / 0.000 | 0.24 / 0.26 / 0.03 / 0.005 | 27–79 | 64 / 320 / 640 / 1280 | – |
| oracle shooting, geodesic cost, H=1 / 5 / 10 / 20 | **1.000** / 0.550 / 0.020 / 0.000 | 1.0 / 0.755 / 0.035 / 0 | 31–60 | 64 / 320 / 640 / 1280 | 0.9–1.4 M |

### What the benchmark does and does not show (be careful)
- MEASURED: a controller with the wall map and true dynamics solves it (100%); greedy and local-sensing reactive controllers solve ≈ 24–30% within the budget; a hand-built retreat/memory heuristic reaches 43.5% given the full 80-step cap but not the locked budget (29.5%). Geometry is tailored to help the heuristic (tangent direction hard-coded to y), so stronger hand-designed heuristics might do better; this was not searched.
- MEASURED: **a one-step lookahead with a map-derived value function is sufficient** (descent, shooting-geodesic H=1). Multi-step *rollout* depth by itself did not help: random shooting with Euclidean cost does not beat greedy at any H tried (it is myopic for the same reason), and the geodesic-cost results *degrade* with H, which reflects the poor sample efficiency of N=64 random-walk plans at long horizons, not the depth needed. The H-sweep is therefore a statement about this sampler, not a clean "required lookahead" curve. A CEM sampler or per-step cost was not tried (out of scope).
- Therefore the defensible claim is narrow: *the task requires non-local (map/value) information that a local reactive policy lacks; a learned planner would need either an accurate multi-step latent rollout with a good search procedure or a learned value/goal-distance head (TD-MPC territory).* It does not show that rollout-based planning is necessary.
- Oracle effort: ~18 simulator steps per decision for descent (≤ 64 for H=1 shooting), plus two grid Dijkstra solves per episode (~3.6 k node expansions each); shooting at H=20 needs 1280 simulator steps per decision.

## 4. Gates (Phase 1)

| Gate | Verdict | Evidence |
|---|---|---|
| **A — Inference correctness** | **PASS** | Executed encoder samples equal logical requests (2560 → 1303/331/313); encoder never called for zero-eligible batches; recurrent state advances when vision is skipped; causal semantics preserved; new `rollout` matches `rollout_legacy` on decisions/success; 24 regression tests + all 17 EXP-0012 predictive-state tests pass (WSL). |
| **B — Measurement integrity** | **PASS** | Evaluation and latency run the same `RecurrentRunner` (equivalence tested, `fixed:K` supported); executed calls and samples counted by instrumentation and independently by a spy; per-component time recorded. Caveat: no like-for-like latency improvement is claimed; bookkeeping overhead at batch 1 is visible (KI-025). |
| **C — Benchmark validity** | **PASS (narrow)** | Oracle solves 100% (200 DEV episodes); local reactive controllers ≤ 30% within the locked budget; walls verified visible and separable at 32×32; planner dynamics equal env dynamics (tested). Narrow because a map-based *one-step* value suffices, so the benchmark does not by itself demonstrate a need for multi-step rollouts (§3). If the founder requires "multi-step rollout necessary", Gate C should be read as **FAIL** for that stronger claim. |
| **D — Scientific readiness** | **PASS, conditional** | Benchmark spec, information table, locked budget, disjoint layout ranges, metrics (success within budget with Wilson CI, finish steps, executed encoder samples/calls, trigger/encoder/update time) and controls are documented here. Still to preregister before a learned-model run: effort-matched baselines (reactive recurrent, 16-step transformer retrained on LA), number of training seeds (≥5; KI-022), the training-time budget (founder approval), a value/goal head decision, and the TEST seeds. |

## 5. Remaining defects and limitations
- KI-025/026 (see KNOWN_ISSUES): EXP-0012's reported encoder counts were logical; instrumented-runner bookkeeping overhead at batch 1.
- Training still encodes every frame (`models.forward_seq`), by design this phase; the EMA target needs all frames.
- Observation status is not exposed to the model (D2); policy still reads the current masked embedding directly (audit §1.3).
- LA v1: no displacement events, deterministic layouts only; blackouts irrelevant to the privileged controllers (they use true state), so occlusion/memory has not been exercised by any LA result yet; wall pixels are drawn over occluders but noise is applied per pixel after drawing.
- Random-shooting oracle results at H ≥ 5 are sampler-limited.
- Fake-quantization findings of EXP-0012 are unchanged; nothing here concerns ternary.
- Several debugging bugs were fixed in the controllers while building them (action-unit scaling; value propagation through wall cells); the reported numbers are from the final code.

## 6. Proposed Phase 2 (NOT implemented) and compute estimate
1. `world_model.py`: belief `b = GRU hidden` with `correct(b, obs)` (existing cell) and `dynamics(b, a, prop_hat)` trained **directly in belief space** (target = closed-loop belief at t+k, stop-gradient), plus small heads: next proprioception (ee, held, grip), target/goal offsets, and a goal-distance (value) head, since §3 shows value information is what planning needs here. Optional JEPA term only as an ablation.
2. Counterfactual dataset from LA env snapshots (K=8 action sequences × horizons ≤10 from sampled states, plus recoveries), generation cost ≈ 10–30 min CPU, numpy.
3. First check the rollout software with the analytic dynamics (`la_move`) and a known linear model (< 1 h, CPU); then one small learned model.
4. `planner.py`: random shooting first (CEM if needed), scoring by the value head only; controls: oracle dynamics, shuffled-dynamics, reactive recurrent, 16-step transformer, retrained on LA.
5. Estimates (not measured): world-model training ~10–20 GPU-min per seed at a reduced budget (above the 15-minute cap → needs your approval per run); 5 seeds × 3 learned arms ≈ 8–12 GPU-hours total, only after Gate B (learned dynamics beat zero-action and action-shuffled controls on held-out layouts). Planner cost to be reported separately: N=64, H=5 ≈ 320 rollout steps per decision, which with the current head sizes is ~30× one encoder pass in MACs, so Gate D (efficient perception) needs the encoder cost stated explicitly.

## 7. Questions for the founder
1. Gate C is narrow. Accept a value-head-based planner as the target ("lookahead + learned value"), or require a benchmark where multi-step rollout without a value function is necessary (e.g. longer corridors, no cheap distance signal)?
2. Decisions D2 (three-state observation code, new model interface) and D3 (belief-space transition) remain open for Phase 2.
3. Approve a first learned run only after Gate B design is preregistered?

## 8. Test commands and results
- WSL (`~/ftvenv`, torch 2.11.0+cu128): `python -m unittest tests.test_predictive_state tests.test_inference_skip tests.test_env_la` → 57 tests OK (17 EXP-0012 + 24 + 16 new). No existing test failed.
- Windows host Python 3.12 (torch 2.11.0+cu128 imported successfully this session, so KI-005 appears no longer to apply on this machine; not investigated): `python -m unittest tests.test_inference_skip` → 24 OK; full run of all 18 test modules → **356 tests OK, 17 skipped** in 1468 s (previous baseline 316 OK). New regressions: none.
- Reproduce: `python -m analysis.predictive_state.run_la_benchmark calibrate|evaluate OUT.json`; `python -m analysis.predictive_state.phase1_measure CKPT OUT.json cuda` (checkpoints are in WSL `~/ps_ckpt`, not in the repo).
