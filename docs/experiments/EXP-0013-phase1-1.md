# EXP-0013 Phase 1.1 — Scientific corrections, fair baselines, planning-benchmark validation

> **Correction notes (2026-11-07, EXP-0013 Phase 1.1 closeout; see `EXP-0013-phase1-1-closeout.md`; the text below is otherwise unchanged and was accurate as written at the time).**
> 1. *Merge status:* this record says the branch is unmerged; `exp-0013-predictive-planning` and `exp-0012-predictive-state` were merged into `main` at `6cbd44b`.
> 2. *Pairing:* the `paired.json` comparisons were **not** fully paired on exogenous events (shared RNG; blackout schedules differed between controllers in 163-193/200 LA_NOM and 195-200/200 LA_OCC episodes, MEASURED). The McNemar p-values in this record are historical and must not be cited as controlled paired tests; corrected paired results (obs_B 200/200 in all three conditions, obs_A 0.995 / 0.970 / 0.560) are in the closeout record.
> 3. *Repeated layouts:* the 200 episodes use 87 unique layouts; episode-level Wilson intervals do not measure uncertainty over unseen layouts (layout-clustered intervals are in the closeout record).
> 4. *TEST split:* the statements "TEST layouts untouched" / "no Phase 1.1 script references the test range" are wrong as worded: `phase11_timing.py` recorded its latency traces on the TEST layout range with the EXP-0012 environment (timing only; no success metric, tuning or claim depends on it). OccPick2D-LA itself was never run on TEST.
> 5. *Power state:* the timing was probably on battery; an AC power repeat (host-reported AC before and after, TRAIN trace) confirms the qualitative conclusions (closeout section 9).
> 6. *Planning wording:* "one-step descent" on the exact geodesic field is a global value-function policy, not a purely reactive one; the Phase 1 anchor confounded horizon with search effort (B = 64·H); the deeper-horizon degradation is a sampler/objective effect at those budgets, not evidence that longer rollouts are harmful (closeout section 10).

Date: 2026-11-07 · Branch `exp-0013-predictive-planning` · Inspected checkpoint: `789b5a887fc9099be5e4b040e687ff72ebe2e01a` (HEAD at start; unchanged by anyone else). Authorization: Phase 1.1 only. No Phase 2 training, no GPU training, no sweeps, no merge, TEST layouts (2000-2199) never used. Labels per AGENTS.md §7. Results: `results/predictive_state_phase1_1/`. Phase 1 record (`EXP-0013-phase1.md`) is preserved; where this record disagrees with it, this record supersedes the *interpretation* (section 8), not the data.

## 0. Founder decision memo (read this first)

**What was fixed.**
1. `rec_updates` counted alive rows although the GRU and policy ran on the whole batch (concern confirmed, `infer.py` at 789b5a8). The runner now executes the cell and policy only on alive rows (`select_alive`, default on), and every counter is checked against an independent spy on the executing module (encoder, cell, policy, trigger head, pixel pooling): 10 new tests.
2. The encoder-skipping latency question is explained, not hidden: skipping removes ~60-67% of the model's MACs but gives **no end-to-end speedup** on batch-1 CPU through the shared runner and **none on CUDA**; it helps only on batched CPU (-9 to -19% wall). Cause: tiny-op dispatch overhead (measured), not the encoder.
3. The Phase 1 benchmark comparison was information-mismatched. It is replaced by an explicit information-access matrix and two observation-matched, non-learned baselines.

**What the evidence now says (MEASURED, DEV, 200 episodes per condition; same layouts and spawns for every controller; blackout realisations are NOT shared, see section 5).**
- Greedy 0.24, reactive-with-memory 0.295, **observation-matched map baseline, no learning, no world model: 1.00 (LA_CLEAR), 0.98 (LA_NOM), 1.00 with a hand-coded 2-part memory (LA_NOM, LA_OCC)**; privileged oracle 1.00.
- The Phase 1 gap between reactive (0.26) and oracle (1.00) was **missing map information, not missing planning**: a one-step descent on a map parsed from the 32x32 RGB frame (wall IoU 1.000) closes it entirely.
- Multi-step planning does **not** beat a one-step value controller: with the exact geodesic value, beam search at H=1/5/10/20 = 1.00/1.00/1.00/1.00. Without a value (Euclidean cost, true dynamics, up to 2720 simulated steps per decision, H up to 20) no planner beats greedy by more than noise (best 0.36 vs greedy 0.24, n=80).
- Persistent memory is needed only to bridge blackouts, and a hand-coded last-seen memory of (wall map, target position, goal box) is sufficient (LA_OCC: single-frame 0.615 -> memory 1.00, discordant on layout/spawn pairs 0 vs 77). Both parts are needed (map-only 0.61, object-only 0.58 on TRAIN).

**Does OccPick2D-LA v1 test the intended architecture? No.** Everything that varies is static (walls, goal, target before grasp) or directly observed (end-effector via proprioception). There is no hidden dynamic state and no action-dependent hidden consequence, so neither a predictive/action-conditioned latent nor counterfactual planning is required, and a hand-coded baseline with no learning saturates it.

**Phase 2 recommendation: BLOCKED as specified (learned dynamics + value-head planner on LA v1); MODIFY.** Training a world model on v1 could at best tie a hand-coded baseline (its memory store is ~15 lines; parser + planner ~150); it could not support the thesis. This is a negative result about the *benchmark*, not a refutation of the hypothesis (which v1 simply cannot test).

**Smallest next experiment (software only, no training, ESTIMATE ~1-2 days of engineering):** specify and calibrate an OccPick2D-LA v2 (section 7) with fresh held-out layouts. Be clear about what each variant can show: (A) a periodic hidden gate seen intermittently tests *prediction through blackouts*, but a hand-coded constant-period extrapolator will probably solve it, so it can only show that a learned state matches a trivial baseline; (B) an **action-dependent hidden consequence** (e.g. a switch that opens the divider gate after a delay, state visible only in valid frames) is the smallest variant that needs action-conditioned prediction and counterfactual rollouts, and is the only one that could separate the proposed architecture from memory plus extrapolation. I recommend authorising (B), with (A) as its control. Proceed to any learned model only if the last-seen-memory baseline fails, a privileged predictor solves it, and a hand-coded predictor is implemented and reported as the bar.

## 1. Executed-computation accounting (section 3 of the directive)

Concern verified at `infer.py` (789b5a8): `c.rec_updates += n_alive` while `net.cell_step` and `net.policy` ran on all `n` rows; trigger pooling also ran on all rows. Fixed in `infer.py`:

| Counter | Meaning | How verified |
|---|---|---|
| `active_steps` | alive rows summed over steps (useful work) | by construction |
| `requested` | alive, valid rows whose trigger asked for an image | `== enc_samples` |
| `enc_samples`, `enc_calls` | images encoded; encoder forward invocations | spy on `net.enc.forward` |
| `rec_updates` | cell rows EXECUTED | spy on `net.cell.forward` rows |
| `policy_rows` | policy rows EXECUTED | spy on `net.pol.forward` rows |
| `trigger_rows` | rows through 8x8 pooling (always the full batch; trigger is not gated) | spy on `models.pool_feat` |
| `finished_rows_processed` | finished rows that still went through cell/policy (0 with `select_alive`) | `rec_updates - active_steps` |
| `t_trigger/t_encoder/t_update` | seconds (timing mode) | perf_counter |

Behaviour: with `select_alive` the cell and policy see only alive rows (gather -> compute -> scatter); finished rows' state and previous action are frozen; an all-finished step performs no trigger, encoder, cell or policy work. The batch-1 path is unchanged. `select_alive=False` reproduces full-batch execution with honest counters. Active-row outputs are preserved (allclose 1e-5 vs full-batch for full/fixed/learned triggers; the closed-loop rollouts remain per-episode identical to `rollout_legacy`, 3 triggers). Existing test `test_full_perception_matches_original_open_loop` was changed to compare active rows only (finished-row state is now frozen, a deliberate change).

New tests (`tests/test_executed_work.py`, 10): counters == spies for four triggers on mixed active/finished/blackout batches; full-batch mode counts finished rows as executed; all-finished step touches nothing; all-skipped vision (no encoder call, cell and policy still run, state evolves); blackout rows never reach the encoder under any trigger; selective == full-batch for active rows; **garbage in finished rows leaves active rows' actions, decisions and states bit-identical**; a row finishing mid-run is frozen; state responds to the executed action and proprioception when vision is skipped. Per-row vs batched equivalence was already covered (`test_runner_batched_equals_per_row_runner`).

Not gated (stated, not hidden): the trigger computation (pooling, learned head) runs on all rows; the encoder-skip arithmetic saving is therefore an upper bound on what a gated-trigger implementation could save.

## 2. Encoder skipping: arithmetic vs measured time (section 4)

Setup: checkpoint `G_s0.pt` (FP32), trace-replay design (`phase11_timing.py`): one closed-loop trace per trigger is recorded; legacy and runner variants replay the *same* inputs, interleaved and rotated across 12 repetitions, 3 warm-up episodes discarded, GC off. Hardware: AMD Ryzen 7 260 (16 logical), NVIDIA RTX 5060 Laptop GPU, WSL2 Ubuntu, torch 2.11.0+cu128, 1 and 4 intra-op threads. **Power state: the host reported `BatteryStatus=1` (discharging) when documented after the runs, i.e. the measurements were probably NOT on AC (KI-021 asks for AC). Absolute numbers moved 20-40% between my two runs (`timing_paired_run1.json` vs `timing_paired.json`); only paired within-run differences are used. A repeat on AC is an unexecuted check.** Power settings were not changed.

Arithmetic: per step MACs encoder 1,126,400, GRU 103,680, policy 34,432, trigger head 20,736 (learned only). Encoder = 87.6% of non-trigger MACs, so theoretical MAC saving at encoder fraction f is 0.876(1-f): fixed:0.25 -> 66%, pix -> 58%, learned -> 63%, full (blackouts only) -> 10%. (INFERRED from layer shapes; MACs are not time.)

**Batch-1 CPU, mean ms/step, 12 paired reps** (legacy = lean EXP-0012 scalar-decision step; `runner` = shipped; `runner_nt` = runner without the per-step stats sync):

| trigger (enc frac) | thr | legacy | runner | runner - legacy [95% CI over reps] | runner_nt - legacy |
|---|---|---|---|---|---|
| full (0.88) | 1 | 0.799 | 0.787 | -0.012 [-0.037, +0.012] | -0.065 |
| fixed:0.25 (0.25) | 1 | 0.563 | 0.745 | +0.182 [+0.148, +0.217] | +0.138 |
| pix:0.0199 (0.34) | 1 | 0.646 | 0.822 | +0.176 [+0.148, +0.204] | +0.105 |
| learned:0.1445 (0.28) | 1 | 0.638 | 0.858 | +0.220 [+0.162, +0.278] | +0.160 |
| fixed:0.25 | 4 | 0.671 | 0.959 | +0.289 [+0.215, +0.363] | +0.185 |
| (4-thread rows for full/pix/learned are in `summary.json`) | | | | | |

Batch-1 p50 / p95 ms (per-step medians over reps; `summary.json`):

| trigger | threads | legacy | runner | runner (no stats sync) |
|---|---|---|---|---|
| full | 1 | 0.758 / 0.887 | 0.751 / 0.868 | 0.693 / 0.821 |
| full | 4 | 0.809 / 1.160 | 0.826 / 1.185 | 0.732 / 1.080 |
| fixed:0.25 | 1 | 0.406 / 0.885 | 0.582 / 1.036 | 0.540 / 1.010 |
| fixed:0.25 | 4 | 0.551 / 1.184 | 0.839 / 1.459 | 0.673 / 1.251 |
| pix:0.0199 | 1 | 0.465 / 0.943 | 0.620 / 1.098 | 0.561 / 1.038 |
| pix:0.0199 | 4 | 0.599 / 1.124 | 0.825 / 1.428 | 0.746 / 1.321 |
| learned:0.1445 | 1 | 0.459 / 0.952 | 0.652 / 1.208 | 0.605 / 1.174 |
| learned:0.1445 | 4 | 0.625 / 1.246 | 0.872 / 1.588 | 0.758 / 1.486 |

In the **lean legacy path** skipping does cut batch-1 latency (full 0.80 -> fixed:0.25 0.56 ms, -30%); the shared runner gives back most of it (0.79 -> 0.75, -5%).

Attribution (MEASURED): `torch.profiler` counts top-level aten ops per step, fixed:0.25: legacy 65.7, runner 95.2, runner without stats sync 84.2 (`op_counts.json`); at ~5-8 us per tiny CPU op this is +0.15-0.25 ms, matching the paired difference. Sources: the tensorised trigger decision (`decide`: comparisons, `torch.where`, accumulator, task/slip checks on 1-element tensors, runner timing component trigger 0.20-0.31 ms) and the stats `.tolist()` sync (~11 ops, ~0.05-0.08 ms). Encoder component in skip modes: 0.14-0.19 ms; update (GRU+policy+bookkeeping) 0.34-0.48 ms. The regression is therefore Python/dispatch overhead, not synchronisation of the encoder and not gather/scatter (batch 1 bypasses them). In `full` mode the runner is not slower: legacy pays for pooling it never uses (66-74 vs 64-75 ops).

**Batched evaluation wall time** (full rollout, paired, interleaved; CPU 32 episodes 4 threads x 5 reps; CUDA 64 episodes x 7 reps), change vs `rollout_legacy`:

| device | trigger | gated full-batch | gated + select_alive (default) | encoder samples executed (gated) |
|---|---|---|---|---|
| CPU | full | -8.9% | -9.8% | 1303 (legacy: all rows every step) |
| CPU | fixed:0.25 | -18.9% | -18.1% | 331 |
| CPU | pix:0.0199 | -17.6% | -13.6% | 464 |
| CUDA | full | +8.9% | +11.2% | 2767 |
| CUDA | fixed:0.25 | +4.0% (CI includes 0) | +9.3% | 713 |
| CUDA | pix:0.0199 | +6.7% | +10.8% | 900 |

Per-episode success identical in every paired run. Interpretation (INFERRED): with this small encoder (1.1 MMAC) the GPU is launch/sync-bound, so executing less arithmetic does not shorten the step and the added gather/scatter/index syncs cost 4-11%; `select_alive` adds more syncs (`nonzero`, indexing) and is the slower default on CUDA. On batched CPU the saving is real but 9-19% against a 58-66% arithmetic reduction, because fixed per-op overhead dominates a model this small. **Conclusion: encoder skipping reduces executed encoder samples by 49-87% and executed MACs by the figures above; it improves total measured inference time only on batched CPU; it does not on batch-1 CPU (shared runner) or CUDA. No energy claim is made or supportable.** Open engineering item (not done, to avoid optimising the benchmark): a scalar fast path for the batch-1 trigger would likely recover ~0.1-0.2 ms.

## 3. Information-access matrix (section 5)

Abridged from `analysis/predictive_state/info_matrix.py` (full generated table with all cell qualifiers: `results/predictive_state_phase1_1/info_matrix.json`; tests check that OBS rows have no privileged access and that the observation baselines' code touches only the observation dict). `Y` receives, `N` does not, `img` recovered from the image by the controller, `ray` local ray-cast through the true map, `hand` hand-coded assumption.

| controller | class | RGB | proprio | task | history | memory | full wall geometry | true dynamics | priv. object pos. | layout id | geodesic map |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | PRIV | N | Y (true ee) | Y | N | N | N | N | **Y** | N | N |
| reactive | PRIV | N | Y (true ee) | Y | N | N | ray | N | **Y** | N | N |
| reactive_memory | PRIV | N | Y (true ee) | Y | N | counters | ray | N | **Y** | N | N |
| oracle_descent | PRIV | N | Y | Y | N | N | **Y** | **Y** | **Y** | **Y** | **Y** |
| oracle_shoot (geodesic cost) | PRIV | N | Y | Y | N | N | **Y** | **Y** | **Y** | **Y** | **Y** |
| oracle_shoot (Euclid cost) | PRIV | N | Y | Y | N | N | Y (via dynamics) | **Y** | **Y** | N | N |
| planning_diag planners | PRIV | N | Y | Y | N | N | Y (via dynamics) | **Y** | **Y** | if geodesic | if geodesic |
| **obs_A_single_frame** | OBS | current frame | Y | Y | N | N | img | hand (kinematic rule, parsed-map collisions) | N | N | derived each step |
| **obs_B_map_memory** | OBS | Y | Y | Y | via memory | hand-coded (wall OR-map, last target, goal box) | img | hand | N | N | derived |
| FUTURE learned (Phase 2, not built) | OBS | Y | Y | Y | via state | learned belief | img (learned) | learned | N | N | N unless a value head is trained (label it) |

The earlier "reactive" and "greedy" were never observation-matched (they read true target/goal/ee; reactive also ray-casts the true map). The Phase 1 table is correctly labelled PRIV from here on. Baselines A/B embed hand-coded knowledge a learned agent must acquire (class colours, object radius prior, the action->displacement rule, gripper rule), which favours them.

## 4. Observation-matched baselines (section 5 A, B)

`controllers_obs.py` (numpy; no training). Input: only `{img, valid, prop, task}`.
- Parser: nearest-centroid pixel classes (BG, goal, occluder, wall, end-effector, six object colours); wall mask (end-effector holes next to a wall filled), target centre by **template fit of the known object radius** (the end-effector disc overdraws the target, a crescent centroid is biased), goal = bounding-box centre of goal pixels.
- Planner: 32x32 occupancy (wall mask dilated by DIL px), 8-connected shortest-path field from the waypoint, one-step descent over 16 headings + direct + toward-best-cell, candidates colliding with the parsed map rejected, final approach direct when the straight line is clear; expert gripper rule on the *estimated* positions.
- **ObsA** is stateless; on a blacked-out frame it holds still. **ObsB** keeps the raw wall class OR-accumulated over valid frames, the last-seen target, the union goal box; the hole-filled mask is used for planning but never stored (a stored end-effector ghost would thicken walls).
- Only tuned parameter: DIL in {0,1}, selected on TRAIN seed 31337 n=60 LA_CLEAR (DIL=1: 0.933 vs 0.800; `dilation_selection.json`).
- **Defects found during development, on TRAIN only, before freezing:** (1) target estimated by pixel centroid made the controller oscillate around the target (TRAIN success ~0.5 -> 1.0 with the template fit); (2) ObsB's first memory design would have stored the end-effector ghost (pointed out by the advisor review, fixed before any run). The parser accuracy script touched DEV and TRAIN *initial frames* (no controller outcomes) before the freeze. **Freeze commit `36bbf13`; the DEV `paired` run happened after it, once, with no subsequent controller change.** The DEV episodes (seed 5001) were not used by any test.

Parse fidelity from one valid frame (150 initial frames each; `parse_fidelity.json`): wall IoU 1.000 (min 1.000) on TRAIN and DEV; target visible 100%, position error mean 0.006 / p95 0.011; goal found 99.3% / 100%, centre error mean 0.010-0.012 / p95 0.031-0.049; shortest-path length from the parsed map is +0.044-0.049 longer on average (dilation) with |error| p95 0.103.

## 5. Results on identical layouts and spawns (DEV layouts, seed 5001, n=200; success within the locked 65-step budget)

**Pairing caveat (checked, 40 episodes, LA_OCC):** layouts, spawns, goals and targets are identical across controllers, but the environment RNG is consumed differently after the first step (the end-effector disc draws a position-dependent number of noise samples), so *blackout realisations differ between controllers* (ObsA vs ObsB: identical valid-frame sequence over the first 20 steps in only 28% of episodes; median first divergence at step 9-10). The McNemar statistics below therefore pair on layout and spawn only, not on blackout timing; the effect sizes are far larger than that nuisance, but the discordant counts are not a clean within-episode comparison.

| controller (class) | LA_CLEAR (no blackout) | LA_NOM (15% dark steps) | LA_OCC (46% dark steps) |
|---|---|---|---|
| greedy (PRIV) | 0.240 [0.19, 0.30] | 0.240 | 0.250 |
| reactive_memory (PRIV) | 0.295 [0.24, 0.36] | 0.295 | 0.295 |
| obs_A single frame (OBS) | **1.000** [0.98, 1.00] | 0.980 [0.95, 0.99] | 0.615 [0.55, 0.68] |
| obs_B map+object memory (OBS) | **1.000** | **1.000** | **1.000** [0.98, 1.00] |
| oracle_descent (PRIV upper bound) | 1.000 | 1.000 | 1.000 |

Wilson 95% CIs. Exact McNemar on layout/spawn pairs (A-only vs B-only successes): LA_NOM 0 vs 4 (p=0.125), LA_OCC 0 vs 77 (p=1e-23); B vs oracle: 0 vs 0 in every condition. Failures of A are all timeouts (stands still in the dark). Blackout statistics (TRAIN, zero-action rollouts of 65 steps): LA_NOM 15.3% of steps dark, 75% of episodes have a blackout, mean longest 7.8 steps; LA_OCC 45.6%, 96.5%, 21.1 steps. LA_CLEAR is defined in the Phase 1.1 script only; `env_la.LA_CONDS` and the 65-step budget are unchanged. 200 episodes per cell: the CIs above are the uncertainty; differences of a few episodes (e.g. 0.98 vs 1.00) are not resolved.

What must be remembered (exploratory, TRAIN seed 2024, n=100, LA_OCC; `memory_ablation.json`): full memory 1.00; map memory only 0.61; object memory only 0.58; none (= A) 0.61. Both are required; neither alone helps.

Compute effort of the observation-matched baselines (unoptimised pure-Python Dijkstra and parsing; `paired.json`): LA_CLEAR 6392 decisions, 1305 Dijkstra solves (~0.2 per decision; solves are cached per map), 6392 parses, 101 s per 200 episodes = 15.8 ms/decision; LA_NOM 13.6-14.2 ms/decision; LA_OCC 8.3-11.5 ms/decision (blackout steps need no parse). Privileged oracle descent for comparison: 18 simulated steps/decision, 0.6-1.4 ms/decision. None of this is on the latency path of the learned model.

Environment/oracle consistency (`consistency.json`): over 64 TRAIN and 64 DEV layouts x 60 random (including over-length) action steps the max |env position - la_move prediction| = 1.4e-7 (float32). TEST layouts untouched.

## 6. Planning-horizon diagnostic (section 5 D; PRIV planners, DEV LA_NOM)

`planning_diag.py`. Anchor: `PlanDiag(rs_walk, geodesic, terminal, B=64H)` on the Phase 1 episode set (n=200) reproduces Phase 1 exactly: 1.000 / 0.550 / 0.020 / 0.000 for H = 1/5/10/20 (so the diagnostic planner is the Phase 1 planner). Grid at n=80 (CI roughly +-0.10 near 0.5), B = simulated steps per decision (N, beam width, CEM samples derived from B; 4 CEM iterations; all replan every step):

| planner | cost | H=1 | H=5 | H=10 | H=20 | B |
|---|---|---|---|---|---|---|
| random shooting, heading random walk, terminal cost (Phase 1) | geodesic | 1.00 | 0.49 | 0.06 | 0.06 | 320 |
| same, **summed per-step cost** | geodesic | 1.00 | 0.99 | 0.94 | 0.75 | 320 |
| constant-heading plans | geodesic | 1.00 | 0.24 | 0.08 | 0.10 | 320 |
| CEM | geodesic | 1.00 | 0.91 | 0.36 | 0.09 | 320 |
| **beam search (16 headings + direct)** | geodesic | **1.00** | **1.00** | **1.00** | **1.00** | 17-340 |
| random walk / terminal | Euclid | 0.16 / 0.00 | 0.13 / 0.00 | 0.03 / 0.00 | 0.01 / 0.00 | 320 / 2720 |
| random walk / summed | Euclid | 0.16 / 0.18 | 0.23 / 0.23 | 0.25 / 0.24 | 0.25 / **0.36** | 320 / 2720 |
| CEM | Euclid | 0.18 / 0.18 | 0.28 / 0.23 | 0.16 / 0.16 | 0.08 / 0.03 | 320 / 2720 |
| beam | Euclid | 0.18 | 0.19 / 0.18 | 0.18 / 0.18 | 0.18 / 0.16 | 340 / 2601 |

(greedy on these episodes: 0.24 at n=200.) Attribution of the Phase 1 degradation:
- **Candidate-sampling failure, not horizon:** a deterministic beam with the same cost holds 1.00 at every H. Random-walk headings (sigma 0.5 rad) almost never produce a consistent far-reaching plan, so the terminal cost of the best sample is noise.
- **Objective scaling / missing intermediate cost:** summing the per-step cost (dense signal) lifts the same sampler from 0.06 to 0.94 at H=10 (0.75 at H=20); terminal-only cost is the main cause with a sampler this weak.
- **Budget:** with fixed B, N shrinks as H grows (confound in the Phase 1 design); CEM recovers part of it (0.91 at H=5) but not at H=10-20.
- **Receding horizon** is common to all arms and is not the cause (beam is receding too).
- **The only arm that can show a benefit of lookahead with true dynamics but *no* map-derived value is Euclid cost:** no planner exceeds 0.36 (best: random-walk/summed at B=2720, H=20; greedy is 0.24; at n=80 this is roughly 1.4 sigma and would need n>=200 to resolve); most are at or below greedy. I did not tune further, per the directive, nor extend to H>20 or B>2720.
- With the exact geodesic value, H=1 is Bellman-optimal by construction, so those arms cannot demonstrate a horizon benefit; they only measure planner quality.

## 7. Does OccPick2D-LA require predictive memory? (section 6, questions 1-7)

1. **Reconstruct the wall map from one visible frame?** Yes. Wall IoU 1.000 on 300 frames; target/goal within ~0.01.
2. **Solve with a conventional map controller and no learned dynamics?** Yes: obs_A 1.00 (LA_CLEAR), 0.98 (LA_NOM); obs_B 1.00 everywhere.
3. **Deterministic enough that position + visible geometry suffice?** Yes: `la_move` is deterministic (consistency 1e-7); randomness is only spawn, pixel noise and blackouts. The map and goal never change; the target is static until grasped.
4. **Hidden state variables or delayed consequences?** None. The only unobserved quantity is "what the last frame showed", which is static.
5. **Reason about alternative actions or follow a static path?** A static shortest path from a parsed/remembered map suffices (H=1 descent and beam at any H are equivalent with the exact value; section 6).
6. **Do blackouts create a benefit for persistent state?** Yes, a bounded one: single-frame 0.615 -> memory 1.00 in LA_OCC. But the needed memory is trivial and static (last map, target, goal), is fully supplied by a 60-line hand-coded store, and needs no dynamics model; predictive (action-conditioned) state is not exercised because proprioception gives the end-effector exactly.
7. **Gap source?** Missing information (map) for greedy/reactive; nothing remaining for planning or control (obs_A with the oracle's own descent rule = 1.00 when the frame is visible). The shooting degradation is the sampler/objective (section 6).

**Plain statement: v1 does not meaningfully require predictive state, counterfactual planning, or learned dynamics.** The only thing it demonstrates is that a persistent memory beats no memory under long blackouts, which a hand-coded memory already achieves.

**Smallest scientifically defensible redesign (PROPOSAL, not implemented; a new benchmark version, LA v2, with its own calibration, preregistered rules and a fresh held-out layout range, e.g. ids 3000-3199).** Two variants, different purposes:
- **v2-A (control): periodic hidden gate.** A gate in the divider cycles open/closed with a per-episode period and phase and is drawn only in valid frames; blackouts hide it, so the state at arrival time must be predicted. This tests prediction through blackouts, but a hand-coded constant-period extrapolator will probably solve it; it can show at most that a learned state matches a trivial baseline.
- **v2-B (the informative one): action-dependent hidden consequence.** A switch the end-effector must press opens the gate after a delay (or only while a second condition holds); the gate state is visible only in valid frames. Choosing the next action then requires predicting the consequence of alternative actions through unobserved intervals, i.e. action-conditioned dynamics and counterfactual rollouts, which a last-seen memory cannot supply.
Required before any training: (i) the last-seen-memory baseline (ObsB) fails clearly; (ii) a privileged planner with the true schedule/dynamics solves it; (iii) a hand-coded predictor (extrapolator for A; a rule-based switch/delay tracker for B) is implemented and reported as the bar a learned state must beat; (iv) observation-matched baselines frozen and committed before any DEV scoring (as done here), and blackout RNG decoupled from rendering so episodes are truly paired.

## 8. Scientific interpretation and gate status (replaces the Phase 1 reading where they differ)

- Multi-step planning vs one-step value controller: **no evidence of improvement** (beam H=1..20 all 1.00 with exact value; no value-free planner reliably beats greedy). INCONCLUSIVE only for horizons >20 / budgets >2720 with Euclid cost, which I did not run.
- Persistent memory necessary: **yes but only as short-term storage of static facts** under blackouts; a hand-coded store is sufficient. Persistent *predictive* state necessary: **no evidence; v1 cannot test it.**
- Encoder skipping improves total measured inference time: **only on batched CPU (-9 to -19%); no on batch-1 CPU through the shared runner (+0.18-0.29 ms in skip modes), no on CUDA (+4-11%).** Executed encoder samples and MACs are reduced as stated.
- Phase 1 Gate C ("PASS, narrow") is **downgraded**: it validated that the benchmark defeats *privileged-state local* controllers, not that it requires planning or prediction. Phase 1 data are unchanged.
- Inconclusive: whether a *learned* compact state can match ObsB on v1 (untested, no training allowed); timing on AC power; other checkpoints; quantized arms (not timed; no packed kernel exists).

## 9. Limitations and unexecuted checks

Single checkpoint (`G_s0`) and single machine for timing, probably on battery; trigger thresholds are the EXP-0012 DEV calibrations; batched CPU/CUDA timings use 5/7 repetitions (CIs are t-intervals over repetitions); n=80 in the horizon grid (wide CIs); DIL is the only tuned parameter but the parser embeds hand-coded priors (colour code, object radius) and the action-effect rule; `dagger.py` collection not re-run; TEST layouts untouched, no preregistered TEST evaluation run (not authorized); no energy measurement; no multi-seed analysis (nothing trained); shooting horizons beyond 20 and budgets beyond 2720 not run; GPU profile of the CUDA slowdown not taken (cause is INFERRED). The first paired timing run's component split was discarded (it summed only the last episode's counters; fixed and re-run; `timing_paired_run1.json` is kept for its latency numbers only).

## 10. Directive section 8 acceptance checklist

| Requirement | Evidence | Result |
|---|---|---|
| Encoder calls/samples equal independently observed executions | `test_executed_work.test_counters_equal_spies_mixed_active_finished`; `test_inference_skip.TestGatedEncode` (spy on `net.enc.forward`) | PASS |
| Recurrent and policy counters match executed rows | same test (spies on `net.cell`, `net.pol`); `test_full_batch_mode_counts_finished_rows_as_executed` | PASS |
| Finished rows accounted for; cannot change active outputs | `test_all_finished_does_nothing`, `test_finished_rows_cannot_change_active_outputs` (bit-identical), `test_row_finishing_midway_is_frozen` | PASS |
| All-skipped batches make no encoder call; blackouts never encoded | `test_all_skipped_vision_...`, `test_blackout_rows_never_reach_encoder_all_triggers`, Phase 1 `test_none_needed_never_calls_encoder` | PASS |
| Active outputs remain correct | `test_selective_equals_full_batch_for_active_rows`; `TestRolloutEquivalence` per-episode identity vs `rollout_legacy` | PASS (finished-row state is now frozen: deliberate change) |
| Controller information access documented | section 3, `info_matrix.py`, `TestInfoMatrix` | PASS |
| Observation-matched baselines on identical episodes | section 5; `TestObservationOnly` (AST check, observation keys only) | PASS for layouts/spawns; **blackout realisations not shared** |
| Oracle upper bounds separated from deployable baselines | PRIV/OBS classes in the matrix and tables | PASS |
| Planning-horizon comparison with documented budgets | section 6; anchor reproduces Phase 1 exactly | PASS with caveat: beam cost is 17WH so its budgets are not equal across H |
| Env transition == oracle transition | `consistency.json` (1.4e-7); `test_env_la.test_planner_dynamics_equal_env_dynamics` | PASS |
| Test layouts untouched | `TestObservationOnly.test_test_range_never_used`; no Phase 1.1 script references the test range | PASS |
| Multi-step planning vs one-step value controller | section 6 | no evidence of improvement (inconclusive beyond H=20 / B=2720) |
| Persistent memory necessary | sections 5, 7 | short-term static memory yes (hand-coded suffices); predictive state not testable on v1 |
| Encoder skipping improves total measured time | section 2 | only batched CPU |

## 11. Reproduce

WSL (torch, `~/ftvenv`): `python -m unittest tests.test_predictive_state tests.test_inference_skip tests.test_executed_work tests.test_env_la tests.test_controllers_obs` (85 tests, all pass). Full Windows-host run of all 20 test modules (torch imports on the host this session): 384 tests OK, 17 skipped (same 17 as the 356-test Phase 1 baseline, none caused by the new modules), 1863 s. Timing: `python -m analysis.predictive_state.phase11_timing ~/ps_ckpt/G_s0.pt results/predictive_state_phase1_1/timing_paired.json 12` (~5 min) and `... phase11_timing ops ~/ps_ckpt/G_s0.pt .../op_counts.json`. Host (numpy): `python -m analysis.predictive_state.run_phase11_benchmark select_dilation|parse_fidelity|paired|memory_ablation|consistency <OUT.json>` and `horizon <OUT.json> 200 anchor`, `horizon <OUT.json> 80 grid 320`, `horizon <OUT.json> 80 grid_euclid 2720`; then `python -m analysis.predictive_state.analyze_phase11 results/predictive_state_phase1_1` and `python -m analysis.predictive_state.info_matrix OUT.json`. Run lengths: paired 10 min, each horizon part 2-3 min, memory ablation 3.5 min, nothing above 15 min.
