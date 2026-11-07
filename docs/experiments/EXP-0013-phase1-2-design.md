# EXP-0013 Phase 1.2: design and falsification gate for OccPick2D-LA v2

Date: 2026-11-07. Status: **DESIGN ONLY, awaiting founder review. Nothing here is implemented, trained or measured, except the two geometry numbers in section 1 (MEASURED from the unmodified v1 code on TRAIN layouts).** Branch `exp-0013-phase1-2-design` (from `exp-0013-phase1-1-closeout` @ `2810a56`; the closeout is **not merged** and has no recorded merge approval, see the decision memo).

Companion documents: `EXP-0013-phase1-2-literature.md` (prior art), `../reviews/MEMO-2026-11-07-exp-0013-phase1-2-decision.md` (decision memo). Evidence labels follow AGENTS.md section 7.

Question this design must be able to answer: **is there a task, simple enough to simulate exactly, on which carrying action-conditioned predictive state forward measurably beats the best non-predictive controllers, and does that gap survive an honest hand-coded predictor?** The design is built so that the answer can be NO.

---

## 1. What Phase 1.1 established (inputs, not re-argued)

- MEASURED: v1 is solved by a hand-coded map+memory controller (`obs_B_map_memory` 200/200 in all three conditions, descriptive on 87 DEV layouts); the memory it needs is static (walls, last target, goal). v1 cannot test predictive state.
- MEASURED (this design, TRAIN ids 0-399, v1 geometry, unmodified `env_la.geodesic_field`): moving from the left half of the arena to the open gap takes a median of 8.2 steps (p95 18.0); gap to goal takes a median of 12.6 steps (p95 18.3, max 19.4). v1 oracle finish is p50 31 / p95 43 steps (`results/predictive_state_phase1/la_calibration.json`). Blackouts last 5-15 (LA_NOM) or 10-30 (LA_OCC) steps. These set the scale for the delay and window below.
- Carry-forward constraints: episode-keyed exogenous streams (`env_paired.py`), explicit layout ids validated against the split, a reserved held-out range guarded in code, a runner that does not distinguish blackout from chosen skip (KI-031).

## 2. v2 environment specification

v2 reuses v1 geometry (vertical divider at x~0.40, open gap, decoy pocket), the 32x32 renderer, the proprioceptive vector, the grip/pick/release rules, `la_move` collision/sliding dynamics and the blackout recurrence. It adds a **gate** in the open gap and a **switch**. v1 files stay untouched; v2 is a subclass in a new module (section 8).

### 2.1 Added entities

| entity | definition |
|---|---|
| gate | wall rectangle `[xw, open_y, WALL_T, 2*GAP_H]` occupying the open gap; present (blocks) iff `g_t = 0`; drawn in the wall colour when present, absent (gap visible) when open |
| switch S | disc of radius `R_S = 0.06` at `p_S` (left half), drawn in a new colour on every valid frame, always visible; never moves |
| decoy pocket | unchanged from v1 (still a dead end) |

Placement rules (episode-keyed, from the layout id): `p_S` in `[0.08, xw-0.12] x [0.10, 0.90]`; `||p_S - spawn_ee|| >= 0.15`; `||p_S - target|| >= 0.12`; not inside a wall margin; geodesic distance from `p_S` to the gap centre between 0.15 and 0.35 (3-7 steps), so the press-to-gate trip is shorter than the delay.

### 2.2 Hidden state (v2-B)

`h_t = (g_t, phase_t, k_t)` with `phase_t in {IDLE, PENDING, OPEN}`:
- IDLE: `g=0`, no countdown.
- PENDING: `g=0`, `k` = steps until opening (1..DELTA).
- OPEN: `g=1`, `k` = remaining passable steps (1..W).

Constants (preregistered initial values, TARGET): `DELTA = 12`, `W = 8`. One recalibration on TRAIN only is allowed (section 7.4); values are then frozen before any DEV2 or held-out run.

### 2.3 Transition equations

Per step `t`, in this order (all v1 rules otherwise unchanged):
1. **Move.** `ee_t = la_move(ee_{t-1}, d(a_t), walls_with_gate(g_{t-1}))`; the gate is a wall iff `g_{t-1}=0`.
2. **v1 interaction** (grip, pick, release, wrong pick, `t_pick`), unchanged.
3. **Switch event.** `in_S_t = ||ee_t - p_S|| < R_S`; `rise_t = in_S_t and not in_S_{t-1}`.
4. **Gate machine:**
   - IDLE and `rise_t` -> PENDING with `k = DELTA`.
   - PENDING: `k <- k-1`; at `k = 0` -> OPEN with `k = W`, `g = 1`.
   - OPEN: `k <- k-1`; at `k = 0`: if the effector is inside the gate slot, hold `k = 0` and stay open until it leaves (the gate never closes on the effector), then -> IDLE, `g = 0`.
   - `rise_t` during PENDING or OPEN is ignored (no restart, no extension).
5. **Blackout recurrence and observation** exactly as v1 / paired replay.

Consequence (closed form, used as a test oracle): a press at step `s` makes the gate passable for moves at steps `s+DELTA+1 ... s+DELTA+W` (exactly `W` consecutive steps) and not otherwise, unless the effector is in the slot at the end of the window.

**v2-A (control).** No switch dependence. The gate follows an exogenous periodic schedule keyed by `(seed, episode_id)`: period `P = 24`, open for 12 consecutive steps, phase `phi in {0..23}` drawn per episode. `g_t = 1` iff `((t + phi) mod P) < 12`; the same "never closes on the effector" rule applies. The switch is rendered but inert. Deadline slack (section 2.6) is chosen so that waiting at the gate is always sufficient (`P` steps of waiting fit inside the budget), i.e. A is **predictable and tolerant**: last-seen memory plus "go and wait at the gate" should stay competitive. This is a calibration claim to be tested (gate V4), not an assumption.

### 2.4 Observation

`obs_t = (img, valid, prop, task)`, identical to v1: 32x32x3 uint8, gray when `valid=0`, `prop = [x, y, held, grip]`, one-hot task colour. The gate state is **visible only through the wall colour on valid frames**; the switch is visible on valid frames; **there is no timer, press flag or phase in any observation.** Proprioception exposes the effector position exactly (including after a blocked move), which is the only channel that carries the agent's own press history. Blackouts follow v1 (cannot start before step 3, so KI-031 is not triggered). Learned-agent interface requirement (not implemented now): add an explicit `observation_status in {valid, blackout, skipped}` input and re-arm the forced first observation after an invalid frame.

### 2.5 Actions

`a = (dx, dy, grip)` exactly as v1 (`MAX_STEP = 0.05`, grip > 0.5 closes). **Pressing is not a separate action**: entering the switch disc is the action. This keeps the action space unchanged and makes the consequence a delayed, hidden effect of ordinary motion.

### 2.6 Reward, deadline, terminals

- Task: pick the target colour, carry it through the gate, release inside the goal square.
- Return (defined for regret and for any later RL baseline): `r_t = -1/T_D` per step, `+1` on success; success = release in goal at `finish <= T_D`.
- Deadline `T_D = ceil(1.25 x p95(oracle finish on TRAIN))` per variant (rule preregistered now; v1 used 1.5x, which here would leave enough slack for blind retry loops to hide the effect, so the multiplier is fixed in advance rather than chosen after seeing baselines; the **deadline sweep** 1.0x-2.0x is reported so no conclusion rests on one multiplier). Simulator cap `T_CAP = T_D + 15`.
- Terminals: success; `wrong_pick` (pick of a distractor); `timeout` at `t_pick >= T_CAP`. No other failure kind.

### 2.7 Exogenous randomness

Per episode, keyed `(seed, episode_id, stream, t)` as in `env_paired`: spawn, blackouts, render noise (and the v2-A phase). The v2-B hidden trajectory has **no randomness of its own**; it is a deterministic function of the action sequence. Hence two controllers on one episode share every exogenous event, and the only source of hidden-state difference is their own presses.

## 3. Paired cases: identical observation, different hidden state, different optimal action

All three types are constructed by scripted prefixes on the same episode (same layout, spawn, blackout schedule, noise). Prefix controllers only move the effector and hold still (loiter); nothing in a prefix reads the hidden state.

**Twin definition.** States `s, s'` at time `t*` form a twin iff (i) `obs(s) = obs(s')` bit-exact (img, valid, prop, task); (ii) hidden states differ; (iii) the future exogenous streams are identical; (iv) the optimal first decision differs: following `s'`'s optimal action in `s` costs at least `EPS = 3` extra steps or fails, and vice versa; (v) **image-channel equivalence of the histories**: every valid frame in both histories shows the gate closed, so the video stream carries no information about the difference.

| type | observation at `t*` | hidden difference | what each state needs |
|---|---|---|---|
| alpha (pending vs idle) | valid frame, gate drawn closed, effector at the same point near S | `s`: PENDING (`k` = 3); `s'`: IDLE (never pressed) | `s`: head for the gate now. `s'`: press S first |
| beta (open vs pending) | blackout (gray), effector at `q` on the route to the gate | `s`: OPEN (`k` = W-2); `s'`: PENDING (`k` = DELTA-4) | `s`: rush through now, window closes in 6 steps. `s'`: keep approaching and wait at the gate (it opens in 8 steps); turning back to re-press would waste a round trip |
| gamma (late vs early in the window) | blackout, same `q` | both OPEN, `k` = 2 vs `k` = 7 | `k=2`: cannot make it (travel 5 steps), return and re-press. `k=7`: go now |

**What makes them distinguishable over time (the only thing).** The time elapsed since the effector last entered the switch disc, i.e. the agent's own proprioceptive/action history. Construction for beta: prefix 1 enters S at step `t1`, walks to `q`, arrives at `t*`; prefix 2 enters S at `t1' = t* - DELTA - 2`, walks to `q`, loiters. Both effector traces end at `q`, held/grip equal. For (v) to hold, no valid frame may occur in `[press + DELTA, t*]` of either history; the construction picks `t*` inside a scheduled blackout and rejects episodes where this fails. alpha uses a valid frame at `t*` and requires that the visible gate is closed in both (PENDING draws closed). A state-fork utility (deep copy of the env at `t*`, continuing under two action sequences, exogenous streams keyed so both branches see the same future) turns each twin into a labelled counterfactual pair for the prediction metrics.

**Why an observation-only memory policy cannot resolve them.** It sees identical current frames; its memory of past frames is identical in the image channel by (v); only the proprio trace differs, and a memory that stores walls/target/goal/last-seen gate state does not store press events or timers. A controller that **does** store "time since I entered S" is exactly the finite-state/timer predictor C2 below, which therefore must solve the twins. That is the point of the benchmark: the twin shows what information is needed, C2 shows it is sufficient.

v2-A has no twins by design: its gate state is not a function of the agent's history; the validity gate V5 checks that A produces none (regret < EPS for every constructed pair).

## 4. Controllers

All observation-only controllers receive exactly `obs_t` plus their own declared memory, never `env.privileged()`, hidden fields, or `la_move` (leakage harness as in `test_controllers_obs.py`). No controller is deliberately weakened: each gets the best information it is entitled to and its free parameters are tuned on TRAIN with the same budget.

| id | name | information access (summary; full matrix in section 5) | why it might succeed | why it might fail |
|---|---|---|---|---|
| C0 | oracle | true `h_t`, true dynamics, exact-dynamics time-expanded search | defines the achievable optimum; plans press timing around the pick | not realisable; search is discretised, so "oracle" = best known (verified against brute force on small cases) |
| C1 | `ObsB2` map+memory, last-seen | v1 `ObsB` memory (walls, target, goal) plus switch position and last-seen gate state. Knows the rule "entering S eventually opens the gate"; keeps a persistent "S visited" bit only if it can read it from its own proprio *now* (i.e., no history) | on v2-A it should be near-optimal: go to the gate, wait, cross when seen/bumped open. On v2-B it succeeds when blackouts are short or the deadline is slack | cannot tell PENDING from IDLE or from an expired window after a blackout: re-presses or waits wrongly; stale "closed" memory |
| C2 | finite-state timer tracker (hand-coded predictor) | C1 plus a log of "entered S at step u" from its own proprio, and the constants DELTA, W | reconstructs `(phase, k)` exactly from the press log; should match C0 up to planning quality | needs DELTA, W known (given); has no way to cope with unmodelled rule changes (B2, out of scope); if the dynamics were stochastic it would need a belief state |
| C3 | non-predictive stateless value policy `ObsA2` | current frame only, geodesic value field treating the gate as passable iff drawn open; no memory | when frames are valid and the gate is open it crosses at once | blackout frames carry no information: it idles; blind to timing |
| C4 | non-predictive probe-and-retry | proprio and bump feedback only: after picking, drive at the gate; if blocked for `K` steps go press S and return; `K` tuned on TRAIN. Uses haptics (a blocked move is observable in proprio) | strong where the deadline is slack: needs no model, only repeated trials | wastes a round trip (about 2 x 5 steps) plus up to `DELTA` waiting per failed cycle; cannot schedule the press around the pick |
| C5 (specified, not run in Phase 1.2) | learned non-predictive recurrent policy | EXP-0012 arm `G0` (behaviour cloning, no predictor head, no JEPA loss) trained on the same data as C6 | the cleanest learned baseline for "does the predictive objective help" | needs training: **not authorized** |
| C6 (specified, not run) | learned predictive-state model | EXP-0012 arm `G`/successor with explicit `observation_status` | the object of the eventual claim | needs training: **not authorized** |

C1 is "the existing map+memory method lifted to v2 with the minimum change needed to be applicable". C4 is the strongest non-model heuristic I can write; including it prevents the benchmark from winning only against a strawman. Both are tuned (K, wall dilation) on TRAIN only and frozen before DEV2 scoring.

## 5. Baseline information-access matrix

Legend: Y = has it, N = does not, ~ = partial (note). `visible gate` = gate state only when the frame is valid.

| information | C0 oracle | C1 ObsB2 | C2 timer FSM | C3 ObsA2 | C4 probe-retry | C5 learned G0 | C6 learned predictive |
|---|---|---|---|---|---|---|---|
| current frame (visible gate) | Y | Y | Y | Y | N (ignores image after parsing static map once) | Y | Y |
| past frames (static map, target, goal memory) | Y | Y | Y | N | ~ (map from first frame) | ~ (recurrent state) | ~ (recurrent state) |
| last-seen gate state | Y | Y | Y | N | N | ~ | ~ |
| proprio now (incl. blocked-move feedback) | Y | Y | Y | Y | Y | Y | Y |
| own press/action history (entry times into S) | Y | N | **Y** | N | ~ (only "blocked for K steps") | ~ (learned) | ~ (learned) |
| constants DELTA, W | Y | Y (rule only, not timing) | **Y** | N | N | learned | learned |
| true hidden state `(phase, k)` | **Y** | N | reconstructed | N | N | N | N |
| future exogenous schedule (blackouts) | Y | N | N | N | N | N | N |
| privileged geometry (walls, goal) | Y | N (parsed) | N (parsed) | N (parsed) | ~ | N | N |

The two rows in bold are the experiment: C2 differs from C1 by exactly the press log and the timing constants, and C2 differs from C0 by exactly the true hidden state (which it reconstructs). Any C2-over-C1 gap is therefore attributable to *action-history predictive state*, not to better perception or planning. The same planner and parser are used by C1, C2 and C3.

## 6. Simulator test plan (before any result is believed)

Numpy-only fast tests, each asserting that the intended event occurred and, where v1 applies, comparing with v1 from an identical state (pattern of `tests/test_transition_events.py`):

1. **Gate machine vs closed form.** Random press times; realised `g_t`, `phase_t`, `k_t` equal the closed-form schedule; passability exactly `W` steps.
2. **Rule fixtures.** Press -> PENDING -> OPEN -> IDLE; presses during PENDING/OPEN ignored; deferred close with the effector in the slot; closed gate blocks and slides; open gate passes; effector never trapped.
3. **v1 non-regression.** With the gate permanently open and the switch inert, v2 physics equals `LAEnv` step for step on random and scripted actions.
4. **v2-A schedule.** `g_t` equals `((t+phi) mod P) < 12` regardless of actions; phase is a function of `(seed, episode_id)` only.
5. **Exogenous pairing.** Different controllers on one episode have identical spawn, blackout, noise and v2-A phase digests; the v2-B hidden trajectory differs only through presses. Batch size/order independence (as `env_paired`).
6. **Fork/restore.** `deepcopy` at `t*`, identical action tail -> identical trajectory; different tails -> different gate states, identical exogenous streams.
7. **Twin construction.** For each type: bit-identical `obs`, differing hidden state, image-channel-equivalent histories (condition v), and oracle regret of the wrong decision >= EPS; the construction rejects episodes that fail and reports the rejection rate.
8. **Oracle correctness.** Oracle success >= 99% on TRAIN; on small instances its finish equals brute-force BFS over (cell, held, phase, k).
9. **Controller leakage harness.** Controllers C1-C4 are run behind an observation-only wrapper; accessing a privileged field raises.
10. **Placement and splits.** Switch constraints hold; layout ids validated against the split on every path; the held-out range is unreachable from any exploratory entry point (split guard extended with the new reserved name and an executed-default check).
11. **RNG guard.** Any draw outside the keyed streams raises (as `_ExoRng`).

## 7. Evaluation protocol

### 7.1 Splits (no reuse of exposed data)

| split | ids | use |
|---|---|---|
| TRAIN | 0-399 | calibration of DELTA/W (one recalibration), deadline, tuning of K and dilation, oracle checks |
| DEV2 | 1100-1199 (never used by v1 scoring) | selection of the "best non-predictive" controller and freezing of all code; also the place for debugging |
| HELDOUT-V2 | 3000-3999 | **locked.** Never used for tuning, debugging or timing. Guarded in code under a new reserved name; one evaluation after freeze |

v1 DEV ids 1000-1099 are not used (their geometry was examined under v1 mechanics). A 1000-id held-out pool addresses the Phase 1.1 limitation that 200 episodes came from 87 layouts: the held-out run uses 400 episodes with 400 distinct layouts per condition (sampled without replacement), so layouts are the unit of analysis.

### 7.2 Conditions

`{A, B} x {LA_NOM blackouts, LA_OCC blackouts}` (blackout parameters as v1). 400 episodes per cell, every controller on the same episode ids and exogenous streams.

### 7.3 Preregistered metrics

| metric | definition | role |
|---|---|---|
| success@T_D | released in goal by step `T_D` | confirmatory |
| decision regret (steps) | `finish(controller) - finish(oracle)` on the same episode; failures censored at `T_CAP` | **primary** (deadline-free) |
| deadline sweep | success vs multiplier 1.0x-2.0x of p95 oracle finish | robustness of the conclusion |
| prediction accuracy | for forks at snapshot times: accuracy of predicted `g_{t+k}`, `k in {1,2,4,8,16}`, conditional on action continuation; MAE of time-to-open when PENDING; reported by observation status (valid / blackout / long blackout) | for any predictor incl. C2 and later C6 |
| calibration | Brier score and ECE (10 bins) of predicted P(`g_{t+k}`=1); meaningful only for probabilistic predictors (C2 is deterministic and reports accuracy only) | C6 |
| action sensitivity | fraction of forks where truth differs between two continuations and the prediction differs too | detects predictors that ignore actions (shuffled-action control from EXP-0012) |
| planning computation | simulated or predicted steps and node expansions per control step (counted, MEASURED); for a learned model, rollout MACs per decision | reported, not gated |
| encoder utilisation | `enc_samples / active_steps`, with `blackout_rows` and `trigger_skipped_rows` separated (needs the `observation_status` interface for learned arms) | reported |
| inference latency | batch-1 CPU p50/p95 on AC power per the Phase 1.1 protocol; hand-coded controllers' Python time is not comparable to model latency and is labelled so | reported |

### 7.4 Calibration rules (TRAIN only, fixed now)

DELTA=12, W=8 as initial values. A single recalibration is permitted if a validity gate fails *for a structural reason on TRAIN* (e.g., oracle cannot reach 99%); the new values and the reason are recorded, then frozen. No parameter is chosen by looking at C1-C4 results on DEV2 or HELDOUT-V2. If a second recalibration would be needed the experiment stops (NO-GO), because repeated adjustment until baselines lose is the failure mode this protocol exists to prevent.

### 7.5 Statistics

Layout-clustered percentile bootstrap (unit = layout, B=4000) for differences; paired exact McNemar on success; Wilson intervals for rates. Because layouts are unique, clustering is nominal but retained. No unconditional-population claim beyond "layouts drawn from this generator".

## 8. Falsifiable GO/NO-GO criterion (written before any experiment)

### 8.1 Design-validity gates (TRAIN, then DEV2; failing any means the benchmark is not a valid test)

| gate | condition |
|---|---|
| V1 solvable | oracle success@T_D >= 0.99 in A and B (both blackout conditions) |
| V2 information necessity (B) | in B (both blackout conditions, DEV2), the best non-predictive controller among C1, C3, C4 has median regret >= 6 steps above C2 **and** success@T_D >= 0.10 below C2 (same thresholds as GO-1, so the gate cannot pass on DEV2 and then fail for a different reason on held-out) |
| V3 information sufficiency | C2 success@T_D >= oracle - 0.05 in A and B (the needed information is available to a bounded observer) |
| V4 control (A) | in A the best non-predictive controller is within 0.05 success and 3 steps median regret of C2 |
| V5 twins | all constructed twins satisfy conditions (i)-(v); A yields none |
| V6 pairing/tests | section 6 tests pass; exogenous digests identical across controllers |

### 8.2 Decision

- **NO-GO (benchmark)** if V1, V2 or V3 fails after the single permitted recalibration, or V4/V5/V6 fail. Interpretation: this family of hidden-state tasks does not isolate a need for predictive state (V2/V4), or is not solvable from the available information (V1/V3). Do not train; return to the founder with the evidence and the option of a different benchmark or a different thesis.
- **GO-1, "predictive information matters"** (confirmatory, one run on HELDOUT-V2, controllers frozen): in B, for both blackout conditions, (a) median regret(best non-predictive) - median regret(C2) >= 6 steps with bootstrap 95% lower bound > 3; and (b) success@T_D(C2) - success@T_D(best non-predictive) >= 0.10 with 95% lower bound > 0; and in A the V4 bounds hold. "Best non-predictive" is fixed from DEV2, not from the held-out result.
- **GO-1 fails on held-out** -> NO-GO, record as a negative result; do not re-tune.

### 8.3 Evidence ladder: what each result would and would not mean

| claim | evidence needed | settled by Phase 1.2? |
|---|---|---|
| E1 predictive (action-history) information matters | GO-1 (C2 and C0 vs C1/C3/C4) | **Yes, this is the only claim Phase 1.2 can establish** |
| E2 the information is sufficient for a bounded observer | V3 (C2 ~ C0) | Yes |
| E3 a learned recurrent state beats a learned non-predictive one | C6 vs C5, paired, same data | No (training not authorized) |
| E4 a learned model is non-inferior to the hand-coded predictor | C6 >= C2 - 0.03 on held-out (stated now as the future bar) | No |
| E5 a learned model beats a hand-coded predictor | needs a variant where an FSM cannot be written cheaply (per-episode latent timing, several interacting switches, stochastic delays: "B2", out of scope) | No, and B cannot show it: C2 is 30 lines |
| E6 efficiency (encoder utilisation, latency, compute) at matched success | after E3/E4 | No |

**Candid limit.** v2-B is a state-tracking task with simple known dynamics. A GO establishes that carrying a timer from one's own action history is necessary; it does **not** show that a learned world model, JEPA objective or ternary state is needed, because the hand-coded C2 will probably solve it. Its value is as the minimum necessary condition for the FlexTern predictive-state thesis: if even this task does not separate C2 from map+memory, no world-model claim on this simulator family is supportable.

## 9. Smallest implementation to test the hypothesis (Stage 1; no training, no GPU)

Order matters; each step has an abort:

1. `env_v2.py` (subclass of `PairedLAEnv`; gate, switch, v2-A/B modes, fork/restore, twin builder) + tests 1-6, 10, 11. ESTIMATE ~300 lines code, 1-1.5 days.
2. Oracle (time-expanded search over `(cell, held, phase, k)` with the true gate dynamics, v1 geodesic heuristic) + TRAIN calibration of `T_D`. Abort if V1 fails. ESTIMATE ~150 lines, 1 day.
3. Controllers C1-C4 (reuse `controllers_obs` parser/planner; C2 adds the press log) + leakage harness. ESTIMATE ~200 lines, 1 day.
4. TRAIN then DEV2 runs of C0-C4 (software, minutes each); evaluate V2-V6. **Abort here on NO-GO** before any held-out machinery.
5. Freeze; one HELDOUT-V2 run; report GO-1.

Not needed: any neural network, GPU, new dependency, change to `env.py`, `env_la.py`, models, RTL or locked Phase 1.1 results. Prediction/calibration/latency metrics are implemented as scoring code for C2 now and reused by C6 later.

## 10. Risks and open questions for the founder

1. **Circularity.** Choosing DELTA/W so that blind retry loses is partly designing the answer. Mitigations: deadline multiplier fixed in advance, a sweep, the strong C4 baseline, one recalibration, and a control variant A that must show the baselines are fine when prediction is unnecessary. Residual risk acknowledged.
2. **Haptic channel.** Bump feedback in proprio lets C4 probe. If C4 is competitive in B the NO-GO is real information (the task is solvable without a model), not a flaw to patch.
3. **Oracle optimality** is relative to a discretised action set (verified against brute force on small cases); regret is therefore a lower bound on true suboptimality for all controllers equally.
4. **KI-031** must be resolved in the learned-agent interface before C5/C6 (observation status input); not needed for Stage 1.
5. **Reserved ranges.** Adding HELDOUT-V2 to `split_manifest.json` and `split_policy.py` is part of Stage 1 step 1; it is the only edit to locked-adjacent infrastructure, and it only adds a guard.
6. **Decision needed from the founder:** authorize Stage 1 (software only), approve the preregistered criteria in section 8, and state whether the unmerged closeout branch should be merged first (this design branch contains its commits).
