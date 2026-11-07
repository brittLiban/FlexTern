# EXP-0013 Phase 1.2: design and falsification gate for OccPick2D-LA v2 (Revision 3)

Date: 2026-11-07. Status: **DESIGN ONLY, awaiting founder review. Nothing here is implemented in the repository, trained or measured, except (a) the two geometry numbers in section 1 (MEASURED from the unmodified v1 code on TRAIN layouts) and (b) the twin arithmetic in section 3, which was checked by a throwaway exhaustive-search script in the 1-D route abstraction (not committed; its outputs are reproduced in the tables).** Branch `exp-0013-phase1-2-design` (from `exp-0013-phase1-1-closeout` @ `2810a56`; that closeout is **not merged** and has no recorded merge approval).

Revision history is in the appendix (Rev 1 -> 2 -> 3, with the superseded statements kept). Companion documents: `EXP-0013-phase1-2-literature.md`, `../reviews/MEMO-2026-11-07-exp-0013-phase1-2-decision.md`, `../reviews/MEMO-2026-11-07-exp-0013-phase1-2-adversarial.md`. Evidence labels follow AGENTS.md section 7; **DERIVED-IN-ABSTRACTION** means proven by exhaustive search in the stated 1-D model only; **HYPOTHETICAL** means not yet executed in the real simulator and not to be treated as validated.

Question this design must be able to answer: **is there a task, simple enough to simulate exactly, on which timing prediction from one's own action history measurably beats the best non-predictive controllers, including a strong press-travel-wait heuristic?** It is built so that the answer can be NO.

---

## 1. Inputs from Phase 1.1 (not re-argued)

- MEASURED: v1 is solved by a hand-coded map+memory controller (`obs_B_map_memory` 200/200 in all three conditions, descriptive on 87 DEV layouts); its memory is static. v1 cannot test predictive state.
- MEASURED (TRAIN ids 0-399, unmodified `env_la.geodesic_field`): left half of the arena to the open gap takes a median 8.2 steps (p95 18.0); gap to goal a median 12.6 (p95 18.3, max 19.4). v1 oracle finish is p50 31 / p95 43 (`results/predictive_state_phase1/la_calibration.json`). Blackouts last 5-15 (LA_NOM) or 10-30 (LA_OCC) steps.
- Constraints carried forward: episode-keyed exogenous streams, explicit layout ids validated against the split, a guarded held-out range, a runner that does not distinguish blackout from chosen skip (KI-031).

## 2. v2 environment specification

v2 reuses v1 geometry, the 32x32 renderer, the proprioceptive vector, grip/pick/release rules, `la_move` and the blackout recurrence, and adds a **gate** in the open gap and a **switch**. v1 files stay untouched; v2 is a subclass in a new module.

### 2.1 Entities and placement

| entity | definition |
|---|---|
| gate | wall rectangle `[xw, open_y, WALL_T, 2*GAP_H]` in the open gap; a wall iff the realized gate state `g = 0`; drawn in the wall colour when closed, absent (gap visible) when open |
| SLOT | the same rectangle regardless of state; `occupied_t = (ee_t in SLOT)` |
| switch S | disc, radius `R_S = 0.06`, at `p_S` in the left half; drawn in a new colour on every valid frame |
| decoy pocket | unchanged |

Placement (episode-keyed): `p_S` in `[0.08, xw-0.12] x [0.10, 0.90]`; `||p_S - spawn|| >= 0.15`; `||p_S - target|| >= 0.12`; outside wall margins; geodesic `p_S` -> gate entry **between 0.25 and 0.35 (G = 5-7 moves)**; **the S disc and the SLOT never overlap, so a press and slot occupancy never coincide**. The lower bound 5 comes from the twin analysis in section 3.4: at G <= 3 no beta/gamma twin exists (pooled regret at most 2) and at G = 4 only marginal ones (regret exactly 3), so a wider placement range would let V5 fail for purely geometric reasons.

### 2.2 Scheduled versus realized gate state (resolves the v2-A contradiction)

Two separate quantities, never conflated:

- **Machine / scheduled state `M_t`**, which does not depend on where the effector is relative to the slot.
  - v2-B: `M_t = (phase_t, k_t)`, `phase in {IDLE, PENDING, OPEN}`, driven only by press events and time: from `M_{t-1}`, if `phase = IDLE` and `rise_t` then `(PENDING, DELTA)`; if `PENDING`, `k <- k-1` and at `k = 0` become `(OPEN, W)`; if `OPEN`, `k <- k-1` and at `k = 0` become `(IDLE, 0)`; a `rise_t` while `phase != IDLE` (evaluated against `M_{t-1}`) is ignored. `open(M_t) = [phase_t = OPEN]`.
  - v2-A: `sigma_t = [((t + phi) mod P) < O]` with `P = 24`, `O = 12`, `phi in {0..23}` drawn per episode from `(seed, episode_id)`; a pure function of `(seed, episode_id, t)`. The switch is rendered but inert.
- **Realized state `g_t`**, the quantity that blocks movement and is drawn:

  `g_t = open(M_t) OR ( g_{t-1} AND occupied_t )`   (the gate never closes on the effector)

  with `open(M_t)` meaning `[phase_t = OPEN]` (B) or `sigma_t` (A).

**Exact order within step `t`** (all other v1 rules unchanged): (1) move: `ee_t = la_move(ee_{t-1}, d(a_t), walls_with_gate(g_{t-1}))`; (2) v1 interaction (grip, pick, release, wrong pick, `t_pick`); (3) `occupied_t`, `in_S_t`, `rise_t = in_S_t and not in_S_{t-1}`; (4) update `M_t` from `M_{t-1}` (B) or compute `sigma_t` (A), using the global step counter `t`; (5) `g_t`; (6) blackout recurrence and observation. Reset: `M_0 = (IDLE, 0)` (B) or `sigma_0` (A), `g_0 = open(M_0)`. The machine and schedule advance only while the episode is live (frozen after `done`). Constants (TARGET, preregistered): `DELTA = 12`, `W = 8`.

**Testable invariants** (each is a named test in section 6):

| id | invariant |
|---|---|
| I1 | no trapping: never (`g_t = 0` and `occupied_t`) |
| I2 | `g_t >= open(M_t)` always |
| I3 | `g_t > open(M_t)` only if `occupied_t` and `g_{t-1} = 1` (held open) |
| I4 | the hold ends the step the effector leaves: if not `occupied_t` then `g_t = open(M_t)` |
| I5 | `M_t` is a function of the rise sequence and time only (B) / of `(seed, episode_id, t)` only (A); it is unaffected by occupancy |
| I6 | B closed form: a press (`rise`) at step `s` with `M_{s-1} = IDLE` makes `open(M)` true after steps `s+DELTA ... s+DELTA+W-1`, so entering moves are possible at steps `s+DELTA+1 ... s+DELTA+W` (exactly `W` of them); a `rise` at step `s+DELTA+W` is ignored, at `s+DELTA+W+1` accepted |
| I7 | if the effector never occupies the slot, `g_t = open(M_t)` for all `t`; in A this is `sigma_t` |
| I8 | A: the `sigma` digest is identical across controllers on one episode; realized `g` digests may differ only at steps with a hold event |
| I9 | machine and schedule are frozen after `done` |

Proof sketch of I1 (induction): suppose `occupied_t` and `g_t = 0`. Then `open(M_t) = 0` and not (`g_{t-1}` and `occupied_t`), so `g_{t-1} = 0`. With `g_{t-1} = 0` the wall was present for the move at `t`; the effector cannot move into a wall rectangle (`la_move`, wall thickness 0.08 > `MAX_STEP`, so no tunnelling), so it was already in the slot at `t-1` with `g_{t-1} = 0`, contradicting I1 at `t-1`. Base case: the spawn is outside every wall margin and left of the slot.

### 2.3 Observation

`obs_t = (img, valid, prop, task)` as v1: gray frame when `valid = 0`, `prop = [x, y, held, grip]`. The gate state is visible **only through the wall colour on valid frames**; the switch is visible on valid frames; **no timer, press flag or phase appears in any observation.** Proprioception exposes the effector position exactly, including after a blocked move (the haptic channel), and is the only channel that carries the agent's own press history. Blackouts follow v1 (none before step 3, so KI-031 is not triggered). A future learned agent additionally needs an `observation_status in {valid, blackout, skipped}` input and a re-armed first observation (not implemented now).

### 2.4 Actions

`a = (dx, dy, grip)` as v1 (`MAX_STEP = 0.05`). Pressing is not an action: entering the S disc is. The consequence is a delayed, hidden effect of ordinary motion.

### 2.5 Reward, deadline, cap, terminals, boundary audit

- Task: pick the target colour, carry it through the gate, release inside the goal square. `finish` = index (1-based) of the step that executes the release. Return (for regret and any later RL baseline): `r_t = -1/T_D` per step, `+1` on success.
- **Inherited simulator cap.** `BatchEnv.step` hard-codes a timeout at `t_pick >= T_PICK = 80`; `LAEnv` adds `t_limit`. In the base step the release (success) is processed before the timeout check, so a release on step 80 counts and nothing after step 80 is possible. **v2 keeps `T_CAP = 80` as inherited, with no override** (so the v1 non-regression test is exact for the whole horizon). Failures are censored at `T_CAP = 80` for regret, which is independent of `T_D`.
- **Deadline.** `T_D = min(75, ceil(1.5 x p95_TRAIN(O_h finish)))`: the inherited v1 multiplier, truncated so that `T_D <= T_CAP - 5`. The truncation is disclosed because it can bind: v1's p95 is 43 and the gate adds up to `DELTA + 1` steps that parallel picking cannot always hide, so the v2 p95 is plausibly 50-60 (ESTIMATE, not measured) and `1.5 x p95` would then exceed 75.
- **p95 versus per-episode guarantees.** The p95 is a statement about the TRAIN distribution (5% of episodes finish later). Success rates need a *per-episode* statement. Requirements, all checked on TRAIN and DEV2 before any controller result is read:
  - R1: `T_D >= ceil(1.1 x max_{TRAIN, DEV2} O_h finish)`.
  - R2: every evaluated episode must have `O_h finish <= T_D`; episodes violating this are labelled `oracle-infeasible`, excluded for **all** controllers symmetrically, listed by id and counted; V1 requires at most 1% of episodes excluded. For the held-out set the exclusion list is produced before controllers are scored.
  - R3 (v2-A): `T_D,A >= p95_A(O_h finish) + (P - O)`, i.e. the worst-case extra wait at a closed gate is the **closed** part of the period (`P - O = 12`), not the whole period; and `T_D,A <= 75`. Chosen from cap arithmetic before any result: v1's p95 is 43, the oracle in A adds at most ~12 steps of closed-gate waiting, so `p95_A <= ~55` and `p95_A + 12 ~= 67 <= 75` holds with margin 8, whereas the rejected `+P` form would give ~79 > 75 and consume the single recalibration before B is calibrated. If R3 still fails on TRAIN the single recalibration may shorten `P` (keeping `O >= W`).
  - R4: the earliest possible finish bounds `T_D` below: `F_min >= 3 + DELTA + 1 + post-gate moves (median 13, max 20)`, i.e. about 29-36 steps before any pick detour; `T_D` must exceed `F_min` for every episode (covered by R2).
- **Terminals:** success; `wrong_pick`; `timeout` at `t_pick >= 80`. No other failure kind.
- **Boundary conditions (each a named test):** E1 entering move at `lo-1`, `lo`, `hi`, `hi+1` of the window; E2 `rise` at `s+DELTA+W` ignored and at `+1` accepted; E3 re-entering the disc while inside creates no `rise`; E4 hold at close, release of the hold when leaving, no re-close while occupied; E5 machine/schedule frozen after `done`; E6 opening hidden by a blackout still obeys the closed form; E7 release on step 80 succeeds, step 81 does not exist; E8 `finish = T_D` is a success, `T_D + 1` a failure; E9 the global step counter, not `t_pick` (which freezes after `done`), drives `sigma_t`; E10 first step: `M_0` and `g_0` as defined, no press possible on step 1 (spawn is >= 0.15 from S).

### 2.6 Exogenous randomness

Per episode, keyed `(seed, episode_id, stream, t)`: spawn, blackouts, render noise, and the v2-A phase `phi`. The v2-B hidden trajectory has **no randomness of its own**. Two controllers on one episode share every exogenous event; the realized gate differs only through their own presses (and, in A, through hold events).

## 3. Counterfactual twins, re-derived

### 3.1 Definitions

A **hypothesis** is a hidden state `h` at time `t*` (machine state plus history that produced it). A **twin** is a pair `(h, h')` with:
- (T1) identical current observation: `obs(h) = obs(h')` bit-exact (img, valid, prop, task);
- (T2) different hidden state;
- (T3) identical future exogenous streams;
- (T4a) **different optimal first decisions**: the sets of optimal first actions in `h` and `h'` (computed by `O_h`) are disjoint; and (T4b) **pooled regret** `rho(h, h') >= EPS = 3` steps, where `rho = min over open-loop action sequences, applied identically until the two observation streams first differ, of max_i (finish_i - V*_i)`, with each hypothesis continuing optimally once its identity is revealed (generous to the pooled policy: a lower bound on the regret of any observation-only policy). `V*_i` is the optimal remaining time in `h_i` (the reference oracle `O_h`, section 4);
- (T5) image-channel equivalence of the histories: every valid frame in both histories shows the gate closed;
- (T6) an **executable witness**: two concrete action scripts from a common start that reach `t*` in `h` and `h'` respectively, replayable in the simulator.

Both T4a and T4b are required (T4a is the founder's criterion; T4b is added because a reversible first move costs one step, so first-action regret with optimal continuation is nearly always ~1 and hides the real cost, namely that an observation-only policy cannot tell which continuation is right). A pair that satisfies T4b but not T4a is **not a twin**; it may still be reported as an *event-memory demonstration pair*.

### 3.2 Route abstraction used for the derivation

1-D integer line; S disc at `x = 0` (a press is the move from `x >= 1` to `x = 0`); gate-entering move is `x: G-1 -> G`, allowed only if `g` was open after the previous step; moves `dx in {-1, 0, 1}`; `held = 1` throughout (the pick is already done, so the post-gate remainder `T_post` is common and cancels in regret). Canonical constants: `DELTA = 12`, `W = 8`, `G = 7` (S->gate is 7 moves, the top of the 5-7 placement range). A waiting position `q` is at `x = G - a` with `a` moves to the entering move and `p = G - a` moves back to the press. The machine is exactly section 2.2. `V*` is computed by exact breadth-first search over `(x, M)`; `rho` by exhaustive search over joint (x, M_h, M_h') open-loop prefixes with divergence on any observation difference (position, or the drawn gate when the frame is valid).

### 3.3 Results (DERIVED-IN-ABSTRACTION; real-simulator status HYPOTHETICAL)

Status key: DIA = derived in the abstraction by exhaustive search; the real-simulator status of every twin is HYPOTHETICAL until Stage 1 test 7 executes its witnesses against the real oracle. Canonical instance `G = 7`, `DELTA = 12`, `W = 8`; numbers are the script's exact outputs.

| pair | position `q` | hidden states | obs at `t*` | optimal first action `h` vs `h'` | T4a disjoint | pooled `rho` (T4b) | verdict |
|---|---|---|---|---|---|---|---|
| **beta** (timing), corrected | `x = 3` (`a = 4`) | `h`: OPEN(k1), `k1 in {1,2,3}` (window closes before arrival); `h'`: PENDING(2) | dark frame | `{-1}` (`{-1,0}` at k1=3) vs `{+1}` | **yes** | 6 (>= 5 further dark frames; 1 frame gives 3) | **twin (DIA, hypothetical in simulator)** |
| **gamma** (timing) | `x = 3` | `h`: OPEN(k1), `k1 in {1,2,3}`; `h'`: OPEN(k2), `k2 in {4..7}` | dark frame | `{-1}` / `{-1,0}` vs `{+1}` | **yes** | 6-7 (>= 3 further dark frames; 0 gives 3) | **twin (DIA, hypothetical in simulator)** |
| **alpha** (event-memory demonstration) | gate-adjacent `x = 6` | `h`: PENDING(k), `k in {5,6}`; `h'`: IDLE | valid frame, gate drawn closed | `{-1,0,+1}` vs `{-1}` | **no** (overlap) | 3 (k=5 or 6) | **not a twin**: fails T4a at every G (below); kept as a demonstration pair for "remembering a press helps"; excluded from V5 yields |

Constraints discovered by the check:
- **alpha fails the founder's criterion.** In the pending state the delay leaves so much slack that "go back and press" is also optimal as a first move, so the optimal first-action sets always overlap; the only cost of the wrong commitment is later. Its `rho` is exactly 3 and only reachable for `k <= DELTA - (G - 1) = 6` (the pending history must have walked from S: `t* - s_press >= G - 1`); `k = 7, 8` would give 4 but cannot occur at that position.
- **beta as written in Revision 2 was not a twin.** It paired OPEN with 6 steps left against PENDING(8) at `a = 4`; both are served by the same action (head for the gate), so `rho = 0` (script output), and its stated press times were inconsistent with a common end position.
- **Condition for any OPEN-versus-other twin:** the window must close before arrival, `k1 < a`; for `k1 >= a`, "go" is optimal in both and `rho = 0`.
- Closed form for the go-then-recover branch: committing to "go" in `h` (`k1 < a`) reaches the closed gate at step `a`, learns it is closed, returns `G - 1` moves and re-presses: extra cost `a + (G - 1) - max(p, k1 + 1)` (7 for `k1 <= 2`, 6 for `k1 = 3` at the canonical constants, equal to the search).

### 3.4 Sensitivity to the switch-to-gate distance G (script, exhaustive; best pair with T4a and `rho >= 3`)

| G | alpha: max rho / T4a | beta: best pair `(q,k1,k2)`, rho; max rho | gamma: best pair `(q,k1,k2)`, rho; max rho |
|---|---|---|---|
| 3 | 1 / none | none; 2 | none; 2 |
| 4 | 2 / none | (1,1,1), 3; 3 | (1,1,3), 4; 4 |
| 5 | 3 / none | (1,1,1), 3; 4 | (1,1,4), 6; 6 |
| 6 | 3 / none | (1,1,1), 3; 6 | (1,1,5), 8; 8 |
| 7 | 3 / none | (1,1,1), 3; 9 | (1,1,6), 10; 10 |

"best pair" is the first one found with T4a and `rho >= 3`; "max rho" is the largest `rho` over all `(q, k1, k2)` for that G (including pairs failing T4a). Reading: **no twins at G = 3, marginal at G = 4, robust from G = 5**, hence the placement range G = 5-7 in 2.1. Alpha never satisfies T4a. The witnesses below use `q = 3` at `G = 7` (rho 6-7); the builder will choose `q` per layout to maximise `rho` subject to witness feasibility.

### 3.5 Executable witnesses (canonical instance, abstraction, verified by the script)

Common start: `x0 = 3` after the pick, `t* = 30`, both scripts loiter (hold still), walk to S, walk back to `x = 3`, loiter.

| twin | script `h` (press step `s`) | script `h'` | hidden at `t* = 30` | dark frames |
|---|---|---|---|---|
| beta k1=3 | loiter 1-10, in 11-13 (press at 13), out 14-16, loiter 17-30 -> OPEN(3) | loiter 1-17, in 18-20 (press at 20), out 21-23, loiter 24-30 -> PENDING(2) | `h`: OPEN(3); `h'`: PENDING(2) | from step 25 (`13 + DELTA`) to >= 35 |
| gamma k1=2,k2=5 | press at 12 -> OPEN(2) | press at 15 -> OPEN(5) | OPEN(2) vs OPEN(5) | from step 24 to >= 33 |
| alpha k=6 (demonstration pair) | `x0 = 6` (gate-adjacent): loiter 1-8, walk in 9-14 (press at 14), walk out 15-20 -> PENDING(6) at `t* = 20` | `x0 = 6`, never press, loiter 1-20 | PENDING(6) vs IDLE | frames valid |

Each witness was replayed in the abstraction: the final position is equal, `M` differs, no valid frame in either history shows the gate open (T5), and `obs(h) = obs(h')` at `t*` (T1). **For the real environment** the twin builder (Stage 1) must (a) generate the same scripts with the layout's true distances, (b) replay both in the simulator and check T1-T5 bit-exactly, (c) compute `V*` and `rho` with the real oracle, and (d) reject the layout/`t*` if `rho < 3` or the blackout does not cover the required frames. Until (a)-(d) have run, every twin is **hypothetical in the real simulator**, and the yield minimums in V5 count only twins that pass (d), and only beta and gamma twins count.

### 3.6 What distinguishes the states, and which controller resolves which

The only distinguishing information is the agent's own proprioceptive/action history (when it entered S). C1 (no press memory) resolves none. C1+ (one remembered "I pressed" bit) resolves the alpha demonstration pair. PTW (section 4) keeps an arrival-wait counter and a patience: it resolves alpha heuristically and can recover from beta/gamma by waiting out its patience at a cost. Beta and gamma require the elapsed time and the DELTA/W model, i.e. forecasting: C2. v2-A has no twins by construction (its schedule does not depend on actions, apart from the hold, which cannot change an optimal decision by 3 steps; V5(c) in 8.1 tests this).

## 4. Controllers and oracles

### 4.1 Oracles (explicit access; neither may use anything outside its declaration)

| id | access | role |
|---|---|---|
| **O_h** (current-hidden-state oracle) | the true current state: effector, objects, held/grip, machine state `M_t` (B) or `phi` and `t` (A), and the true transition function. **Excluded:** future blackouts, future render noise, other episodes, controllers' choices. Plans by exact-dynamics time-expanded search. | **defines `V*` and regret** (the achievable optimum for an agent that knows the state but not the future exogenous draws) |
| **O_c** (clairvoyant) | everything O_h has **plus** the future blackout schedule and noise (and, in any stochastic extension such as B2, the future random draws) | **diagnostic only**: lower bound on perception scheduling (how few frames an ideal encoder policy needs given the blackouts), hindsight gap in stochastic variants. Never a regret reference and never reported as a controller result |

In v2-A and v2-B as specified, blackouts and noise do not act on physics, so O_h and O_c have identical physical finish times (asserted by test); they differ for perception metrics, and would differ for stochastic B2. Enforcement: access wrappers expose only the declared fields, and tests (section 6) check that O_h's actions are invariant under permutation of the future blackout/noise streams and that O_c's extra fields are exactly the declared ones.

### 4.2 Controllers

All observation-only controllers share one parser, one route planner (geodesic descent on the parsed map), one actuation/grip rule and the same observation access, and differ only in a pluggable **gate policy** (a test checks that with the gate always open, or on v1, all of them emit identical actions). None is weakened; free parameters are tuned on TRAIN with equal budgets and frozen before DEV2.

| id | name | memory / information (beyond the shared parser, planner and current observation) | why it might succeed | why it might fail |
|---|---|---|---|---|
| C1 | `ObsB2` map+memory | v1 `ObsB` memory (walls, target, goal) + switch position + last-seen gate state; knows the rule "entering S eventually opens the gate"; no record of presses | v2-A: go to the gate and wait; v2-B: short blackouts | cannot tell PENDING from IDLE or an expired window |
| C1+ | `ObsB2` + event bit | C1 + one persistent flag "I have entered S" (no time stamp, no counter, no DELTA/W) | avoids pointless re-pressing; resolves alpha | cannot decide when to give up or when the window has closed |
| **PTW** (C1++) | **press-travel-wait** | C1+ + a wait counter (steps since arrival at the gate, reset when it leaves) + a scalar patience `Pi` tuned on TRAIN. **No** time-since-press, **no** DELTA/W, no forecast. Behaviour: go press, travel to the gate, wait/bump up to `Pi` steps, pass when open; if the patience expires, return and re-press. **Variants:** *press-after-pick* (pick, press, gate) and *press-before-pick* (press, pick, gate), plus a non-forecasting rule variant (press first iff the target is nearer to S than to the gate entry); the best variant is selected on DEV2, not on the held-out set | strong where blackouts are long and the deadline allows one retry; press-before-pick overlaps the delay with picking without a model; `Pi ~ DELTA+` makes it competitive | the tuned `Pi` implicitly encodes DELTA (disclosed: the line between "tuned patience" and "forecast" is thin, which is exactly why the C2-PTW gap is the honest measure of the model); misses the window when pick+travel is outside `[DELTA+1, DELTA+W]`, then waits `Pi` before recovering |
| C2 | finite-state timer tracker (hand-coded predictor) | C1+ + press time stamps + the constants DELTA, W: reconstructs `(phase, k)` and forecasts `g_{t+j}`; schedules the press relative to the pick (press first iff the forecast arrival falls in the window), decides go/re-press from the forecast. **In v2-A:** gets `P` and `O` and extrapolates the phase from the last gate transition it saw (time since transition mod `P`); with none seen it behaves like C1 | reconstructs the hidden state exactly; should approach O_h up to planning quality | needs DELTA, W given; breaks under unmodelled changes (B2); with stochastic delays would need a belief state |
| C3 | `ObsA2` stateless | current frame only, value field treating the gate as passable iff drawn open | acts immediately when frames are valid and the gate is open | idles through blackouts |
| C4 | bump-only probe-and-retry | proprio/bump feedback only; no press bit; after `K` blocked steps it goes to S | needs no model | wastes round trips; cannot schedule |
| C5 / C6 | learned non-predictive / predictive recurrent models | EXP-0012 `G0` / `G`-successor with explicit `observation_status` | the eventual object of study | training **not authorized**; specified only |

The **non-predictive set** used in V2 and GO-1 is `{C1, C1+, PTW (best variant), C3, C4}`. A GO therefore means that forecasting beats the strongest forecast-free controller that has the same parser, planner and observation access, not merely that remembering a press helps.

## 5. Baseline information-access matrix

Y = has it, N = does not, ~ = partial. Gate state "visible" means on valid frames only.

| information | O_h | O_c | C1 | C1+ | PTW | C2 | C3 | C4 | C5/C6 (learned) |
|---|---|---|---|---|---|---|---|---|---|
| current frame (visible gate) | Y | Y | Y | Y | Y | Y | Y | N (ignores image after parsing the static map) | Y |
| static map, target, goal memory | Y | Y | Y | Y | Y | Y | N | ~ | ~ |
| last-seen gate state | Y | Y | Y | Y | Y | Y | N | N | ~ |
| proprio now, incl. blocked-move feedback | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| "I entered S" event bit | Y | Y | N | Y | Y | Y | N | N | ~ |
| wait counter since arrival at the gate; patience `Pi` | n/a | n/a | N | N | **Y** | Y | N | ~ (`K`) | ~ |
| press time stamps + gate model (DELTA, W) -> forecast | Y | Y | N | N | N | **Y** | N | N | ~ (learned) |
| true hidden state `M_t` (or `phi`) | **Y** | Y | N | N | N | reconstructed | N | N | N |
| future blackouts / noise | N | **Y** | N | N | N | N | N | N | N |
| privileged geometry | Y | Y | N (parsed) | N | N | N | N | ~ | N |

The experiment sits on two rows: **PTW vs C2** differ by exactly the time stamps and the gate model (forecasting), so a C2-over-PTW gap measures the value of the model; **O_h vs C2** differ by exactly the true state (which C2 reconstructs). The same parser and planner are used by C1 through C3 and C4.

## 6. Simulator test plan (before any result is believed)

Numpy-only fast tests; each asserts the intended event occurred and, where v1 applies, compares with v1 from an identical state (pattern of `tests/test_transition_events.py`):

1. **Gate machine vs closed form** (I5, I6, I7, E1, E2): random press times against the section 2.2 closed form; window of exactly `W` entering moves.
2. **Invariants I1-I4, I9** with adversarial action scripts that dwell in the slot across the scheduled close, leave, and re-enter; **E3-E5, E10**.
3. **v1 non-regression for the whole horizon `T <= 80`** with the gate permanently open and the switch inert.
4. **v2-A schedule and scheduled/realized split** (I2, I3, I7, I8): `sigma_t` equals the formula for any action sequence; realized `g` equals `sigma` unless held.
5. **Exogenous pairing:** identical spawn, blackout, noise and `phi` digests across controllers; batch size/order independence.
6. **Fork/restore:** `deepcopy` at `t*`, same tail -> identical; different tails -> different gate states, identical exogenous streams.
7. **Twin construction (the gate for section 3):** replay both witness scripts in the real simulator; check T1-T5 bit-exactly; compute `V*` and `rho` with O_h; reject if `rho < 3`; report rejection rates. **Until this passes, twins are hypothetical.**
8. **Oracle correctness and access:** O_h success >= 99% on TRAIN; equals brute-force BFS on small instances; O_h actions invariant under permuting future blackout/noise streams; O_h and O_c physical finish identical; access wrappers expose exactly the declared fields.
9. **Controller leakage harness** for C1-C4 and PTW; **shared-pipeline test** (identical actions when the gate policy is neutral).
10. **Deadline/boundary tests** E6-E9, R1-R4 computed on TRAIN/DEV2 and written to the calibration file before any controller runs.
11. **Placement, splits and RNG guard:** S/slot disjointness, layout ids validated against the split, the held-out range unreachable from exploratory entry points (extend the existing guard and its executed-default test), any draw outside the keyed streams raises.

## 7. Evaluation protocol

### 7.1 Splits

| split | ids | use |
|---|---|---|
| TRAIN | 0-399 | calibration of `DELTA`/`W` (one recalibration), `T_D`, tuning of `K`, `Pi`, wall dilation, variant choice, oracle checks |
| DEV2 | 1100-1199 (never scored under v1) | selection of the best non-predictive controller and freezing of code; debugging |
| HELDOUT-V2 | 3000-3999 | **locked**; never used for tuning, debugging or timing; guarded in code; one evaluation after freeze |

400 held-out episodes per cell on 400 distinct layouts, so layouts are the unit of analysis (addresses the Phase 1.1 limitation). Conditions: `{A, B} x {LA_NOM, LA_OCC}`; every controller on the same episode ids and exogenous streams.

### 7.2 Preregistered metrics

| metric | definition | role |
|---|---|---|
| decision regret (steps) | `finish(controller) - finish(O_h)` on the same episode; failures censored at `T_CAP = 80` | **primary, deciding** (deadline-free) |
| success@T_D | released in the goal by step `T_D` (rule in 2.5) | reported; criterion in 8.2 (P3) |
| deadline sweep | success vs deadline from 1.0x p95 to `T_CAP` | robustness |
| prediction accuracy | accuracy of predicted `g_{t+k}` (`k` in {1,2,4,8,16}) conditional on an action continuation, on forked snapshots; MAE of time-to-open when PENDING; by observation status | any predictor (C2 now, C6 later) |
| calibration | Brier and ECE (10 bins) of predicted P(`g_{t+k} = 1`); meaningful for probabilistic predictors only | C6 |
| action sensitivity | fraction of forks where truth differs between two continuations and the prediction differs too | detects action-blind predictors |
| planning computation | simulated/predicted steps and node expansions per decision (counted); MACs for a learned model | reported |
| encoder utilisation | `enc_samples / active_steps`, with `blackout_rows` and `trigger_skipped_rows` separated | reported; O_c gives the lower bound |
| inference latency | batch-1 CPU p50/p95 on AC power (Phase 1.1 protocol); hand-coded Python times are not comparable and are labelled so | reported |

### 7.3 Calibration rules (TRAIN only)

`DELTA = 12`, `W = 8` as initial values. **One** recalibration is allowed, only for a structural reason (R1-R3 failing, O_h unable to reach 99%, or `T_D` untenable under the 80-step cap); the values and reason are recorded, then frozen before DEV2 or held-out runs. No parameter is chosen by looking at C1-C4/PTW results. A second recalibration means stop (NO-GO).

### 7.4 Statistics

Layout-clustered percentile bootstrap (B = 4000) for paired differences; exact McNemar for success; Wilson intervals for rates. No claim beyond "layouts drawn from this generator". The paired regret SD is estimated on DEV2 for **reporting power only**; it is never used to change a threshold. If more than half of one arm's episodes time out, that arm's median regret is determined by the 80-step cap rather than by behaviour; the timeout fraction is reported for every arm and, in that case, the mean-regret difference is reported alongside the median.

## 8. Falsification criteria

### 8.1 Gate table: Revision 2 as registered, and Revision 3 as proposed (row by row)

"Best non-predictive" is fixed on DEV2 and never re-selected on held-out data. "B" has two blackout conditions, LA_NOM and LA_OCC. `NP-set(Rev2) = {C1, C1+, C3, C4}`; `NP-set(Rev3) = {C1, C1+, PTW (best variant), C3, C4}`.

| gate | Revision 2 (registered, as reviewed) | Revision 3 (proposed; P-numbers in 8.2) |
|---|---|---|
| V1 solvable | oracle success@T_D >= 0.99 in A and B (both blackout conditions) | `O_h` per-episode feasibility: at most 1% `oracle-infeasible` episodes in each cell, R1-R4 satisfied |
| V2 information necessity (B, DEV2) | in both blackout conditions: best of NP-set(Rev2) has median regret >= 6 steps above C2 **and** success@T_D >= 0.10 below C2 | NP-set(Rev3); median regret gap >= 6 with lower bound > 3 (P1, unchanged); success criterion replaced by non-inferiority (P3); overall outcome is the weaker of the two blackout conditions (P7) |
| V3 sufficiency | C2 success@T_D >= oracle - 0.05 in A and B | C2 median regret vs `O_h` <= 3 steps **and** success within 0.05 (P5) |
| V4 control (A) | best of NP-set(Rev2) within 0.05 success and 3 steps median regret of C2 (C2 as defined for A) | NP-set(Rev3); same bounds |
| V5 twins | (a) every constructed twin satisfies conditions (i)-(v); (b) minimum yield on DEV2 (full 100-layout pool, several `t*` per episode; TARGET): alpha >= 100 per blackout condition, beta and gamma >= 60 each in LA_OCC (LA_NOM yield only reported, since 5-15-step blackouts rarely cover the opening); (c) A yields none (regret < EPS for every constructed pair); failing the yield means the condition is untestable, not that the minimum is relaxed | (a) every twin satisfies T1-T6 incl. T4a (disjoint first actions) and T4b (`rho >= 3`) **validated with the real oracle** (test 7); (b) **alpha removed** (fails T4a); beta and gamma >= 60 each in LA_OCC from validated twins, placement G = 5-7, remaining-dark requirements (beta >= 5, gamma >= 3 frames); (c) unchanged |
| V6 pairing / tests | section 6 tests pass; exogenous digests identical across controllers | same, plus tests 7-10 |
| GO-1 (one run on HELDOUT-V2, controllers frozen) | in B, for both blackout conditions: (a) median regret(best of NP-set) - median regret(C2) >= 6 steps, bootstrap 95% lower bound > 3; (b) success@T_D(C2) - success@T_D(best of NP-set) >= 0.10, lower bound > 0; and in A the V4 bounds hold | partition on the lower bound (below); (b) replaced (P3) |
| recalibration | one, TRAIN only, structural reasons only; a second means stop | unchanged |

**Outcome partition (Revision 3 proposal, closing the gap between GO and SMALL-POSITIVE).** For each blackout condition, let `est` be the paired median-regret gap (best of NP-set minus C2) and `LB` its bootstrap 95% lower bound:
- **GO**: `est >= 6` **and** `LB > 3`, and the non-inferiority criterion (P3) holds, and the A-control bounds hold;
- **SMALL-POSITIVE**: `LB > 0` but not GO;
- **NO-GO**: `LB <= 0`, or any V-gate failure after the single recalibration.
The overall outcome is the **weaker** of the LA_NOM and LA_OCC outcomes (P7), which keeps the Rev 2 requirement that both conditions pass. Because the twin analysis predicts timing twins mostly in LA_OCC, an LA_NOM shortfall is a likely way to land in SMALL-POSITIVE; that is accepted, not engineered around.

### 8.2 Review after adding PTW and the oracle split (PROPOSED changes; nothing has been run)

| # | item | Rev 2 | Rev 3 proposal | reason |
|---|---|---|---|---|
| P0 | non-predictive set | `{C1, C1+, C3, C4}` | add PTW (best variant) | the strongest forecast-free controller must be in the set; **applied regardless of approval** |
| P0' | regret reference | "the oracle" | `O_h` | access declaration (4.1); **applied regardless** |
| P1 | regret margin | 6 steps, lower bound > 3 | **unchanged** | set before PTW existed and not derived from any baseline result; lowering it after strengthening the baseline would be moving the goalposts. Noise check: regret is censored at 80 and the oracle finish is >= ~29, so a per-episode regret can span ~50 steps; the paired SD is therefore at most ~25 (an assumption, not a measurement), SE <= 1.25 at n = 400, and `LB > 3` effectively requires `est` of about 5.5 or more, consistent with the 6-step threshold |
| P2 | small-positive band | none | the SMALL-POSITIVE outcome above (lower bound > 0, not GO): reported as evidence that timing helps modestly; treated as NO-GO for authorizing learned-model work | with a strong PTW a small residual gap is the likely honest outcome; the band prevents it being relabelled after the fact |
| P3 | success criterion | `>= 0.10` and lower bound > 0, conjunctive | **non-inferiority**: `success(C2) >= success(best of NP-set) - 0.02`; full sweep reported | with a 1.5x (truncated) deadline a strong PTW may succeed on ~100%, so a conjunctive success margin could return NO-GO for a ceiling reason unrelated to the hypothesis; regret is the deciding deadline-free metric. **This relaxes a condition (it favours GO); P1's unchanged margin, P2 and the stronger set offset it** |
| P4 | V2 vs GO-1 | same thresholds | same thresholds | a gate must not pass on DEV2 and fail on held-out for a different reason |
| P5 | V3 | success within 0.05 | add median regret <= 3 steps | success can saturate |
| P6 | V4 | within 0.05 / 3 steps | same with NP-set(Rev3) | unchanged in substance |
| P7 | both blackout conditions | both required | weaker-of-two rule | formalises the Rev 2 requirement under the partition |

Operative status: **until the founder approves Revision 3, the Rev 2 column of 8.1 stands, with P0 and P0' applied**. Nothing is executed before approval.

### 8.3 Evidence ladder: what each result would and would not mean

| claim | evidence | settled by Phase 1.2? |
|---|---|---|
| E1a remembering one's own action matters | C1+ vs C1, C3, C4 | yes (reported) |
| E1b timing **forecasting** from action history matters | GO-1: C2 vs the best of C1, C1+, PTW, C3, C4 | **yes: the only predictive claim Phase 1.2 can establish** |
| E2 the information is sufficient for a bounded observer | V3 (C2 ~ O_h) | yes |
| E3 a learned recurrent state beats a learned non-predictive one | C6 vs C5, paired, same data | no (training not authorized) |
| E4 a learned model is non-inferior to the hand-coded predictor | C6 >= C2 - 0.03 on held-out (future bar) | no |
| E5 a learned model beats a hand-coded predictor | needs a variant where an FSM cannot be written cheaply (B2 below) | no, and B cannot show it |
| E6 efficiency (encoder use, latency, compute) at matched success | after E3/E4 | no |

## 9. Why v2-B establishes only a prerequisite claim, and a sketch of B2

**Why prerequisite only.** In v2-B the dynamics are deterministic, known and tiny (a three-phase counter with two constants). A hand-coded finite-state tracker (C2, ~30 lines) can solve it, so a GO shows that *carrying a timer derived from one's own actions is necessary* and that an observer with that information is sufficient; it cannot show that a learned world model, a JEPA objective, ternary state or encoder skipping is needed or beneficial. Specifically it does not test: (1) learning dynamics from data (nothing is unknown to the designer), (2) uncertainty (no belief state is needed), (3) perception cost (no frame is more informative than another; there is nothing to schedule), (4) generalisation to unmodelled consequences. It is the **minimum necessary condition** for the predictive-state thesis on this simulator family: if even this task does not separate C2 from the strongest forecast-free controller, no world-model claim here is supportable; if it does, the claim is only that the thesis has not been ruled out.

**B2 concept (SPECULATIVE; not specified in detail, not to be implemented).** Make the consequences uncertain and the observations selectively informative:
- *Uncertain consequences:* the press delay is drawn per episode from a small set (e.g. {8, 12, 16}) or per press, and is revealed only by watching one full cycle; several switches interact (e.g. one shortens the delay, one closes the gate); the window `W` varies.
- *Selectively informative observations:* a cheap low-bandwidth channel (proprio, bump feedback) is always available, while the full-frame encoder costs compute; the frame is informative mainly in short intervals (the gate becomes visible/flickers shortly before it opens, or a state lamp is legible only in a narrow window), and uninformative otherwise.
- *What this makes meaningful:* the agent must hold a *belief* over the delay class, update it from sparse events, and choose **when to spend perception** (look when the posterior says the opening is imminent or uncertain; skip when confident). A fixed-patience heuristic then pays either wasted waiting or missed windows; the value of information and the encoder-utilisation/latency metrics become non-trivial, and the stochastic draws make O_h (state known, draws unknown) differ from O_c (hindsight), so regret against O_h and the O_c gap measure different things. This is where a learned recurrent state could plausibly beat a hand-coded tracker, and where adaptive perception has something to decide. B2 is a separate objective that needs its own preregistration and authorization.

## 10. Smallest implementation to test the hypothesis (Stage 1; no training, no GPU)

1. `env_v2.py` (subclass of `PairedLAEnv`: machine/schedule, hold rule, switch, fork/restore, twin builder) + tests 1-6, 11. ESTIMATE ~350 lines, 1-1.5 days.
2. O_h search (time-expanded over `(cell, held, M)`, v1 geodesic heuristic) + TRAIN calibration of `T_D` with the R1-R4 checks; abort if V1 fails. ESTIMATE ~150 lines, 1 day.
3. Twin builder and test 7 (**abort if no layout yields twins that pass T1-T5 with `rho >= 3`**); controllers C1-C4, PTW, C2 on the shared pipeline + leakage and shared-pipeline tests. ESTIMATE ~250 lines, 1-1.5 days.
4. TRAIN then DEV2 runs; evaluate V1-V6; **abort on NO-GO before any held-out machinery**.
5. Freeze; one HELDOUT-V2 run; report GO-1 / SMALL-POSITIVE / NO-GO.

Not needed: any neural network, GPU, new dependency, change to `env.py`, `env_la.py`, models, RTL or locked Phase 1.1 results.

## 11. Open risks and decisions for the founder

1. **Circularity** remains the central risk: delay/window sizes and the deadline are designer choices. Mitigations: inherited multiplier, regret as the deciding metric, strong PTW, control variant A, one recalibration. The stronger PTW makes NO-GO more likely; that is the point of the gate.
2. **The abstraction is not the simulator.** Section 3 is exact only in the 1-D collinear model; real layouts change distances and may remove or create twins. Treat every twin as hypothetical until test 7.
3. **Tuned patience hides a model.** PTW's `Pi` approximates DELTA. The design reports the C2-PTW gap as the value of the model, not as proof that no model is present in PTW.
4. **Thresholds** P1-P6 need approval before implementation (Q-029). The 80-step cap truncates `T_D` and may bind.
5. **KI-031** must be resolved in the learned-agent interface before C5/C6.
6. **Decisions needed:** approve or amend section 8.2; authorize Stage 1; order of the pending closeout merge and RTL chunk; held-out ids 3000-3999 (Q-029 to Q-032).

---

## Appendix: revision history (superseded statements preserved)

**Revision 1** (first draft, commit `b7b67a7`). Gate machine had the hold rule inside the machine (`OPEN, k = 0` hold), so v2-A (an exogenous periodic schedule) and "the gate never closes on the effector" contradicted each other. Twins alpha/beta/gamma were stated as: alpha at the switch (pending vs idle), beta "open (k = W-2) vs pending (k = DELTA-4)", gamma "k = 2 vs k = 7 in the window"; deadline `T_D = ceil(1.25 x p95)` with `T_CAP = T_D + 15`; controllers C0-C6 without an event-memory baseline.

**Revision 2** (commit `5e042f0`, after advisor review `../reviews/REV-2026-11-07-exp-0013-phase1-2-design-review.md`). Added C1+ (event bit); deadline rule changed to the inherited 1.5x with a deadline-free deciding metric, `T_CAP = ceil(2 x p95)`; C2 in v2-A specified; twin yields preregistered; criteria V1-V6/GO-1 with 6 steps / 0.10 / 0.05.

**Revision 3** (this file; review `../reviews/REV-2026-11-07-exp-0013-phase1-2-correction-review.md`). (1) Scheduled and realized gate states separated, exact order and nine invariants (I1-I9). (2) Twins re-derived with pooled regret and exhaustive search in a stated abstraction; **Rev 1/2 alpha (at the switch, harmless re-press) and beta (both states best served by "go") were not valid twins and are replaced**; every twin labelled DIA or HYPOTHETICAL. (3) PTW (press-travel-wait, press-before/after-pick) added to the non-predictive set. (4) Oracles split into O_h (regret reference) and O_c (diagnostic) with access declarations and tests. (5) Boundary audit: `T_CAP = 80` inherited, `T_D` truncated at 75, per-episode requirements R1-R4 and boundary tests E1-E10; the Rev 2 `T_CAP = ceil(2 x p95)` is withdrawn because it could exceed the simulator's 80-step timeout. (6) Section 9 added (prerequisite-only claim; B2 sketch). (7) Threshold proposals P0-P6 recorded without change to the registered Rev 2 criteria until approved. (8) After an independent review of Rev 3 (`../reviews/REV-2026-11-07-exp-0013-phase1-2-correction-review.md`): full gate table restored with Rev 2 and Rev 3 columns; **alpha reclassified as not a twin** (fails the first-decision criterion at every G) and removed from the V5 yield; outcome partition closed on the lower bound with a weaker-of-two-conditions rule; R3 corrected to `+(P - O)`; placement narrowed to G = 5-7 from the G-sensitivity sweep; the SD assumption corrected to ~25. Independent review and adversarial memo (`../reviews/MEMO-2026-11-07-exp-0013-phase1-2-adversarial.md`) added.
