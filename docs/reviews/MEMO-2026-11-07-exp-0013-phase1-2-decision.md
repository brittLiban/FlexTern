# MEMO 2026-11-07: Phase 1.2 decision (design and falsification gate)

To: founder (for discussion with ChatGPT). From: implementation agent. Status: recommendation; nothing implemented.

## Repository and approval state

- This work is on `exp-0013-phase1-2-design`, created from `exp-0013-phase1-1-closeout` @ `2810a56`.
- **The Phase 1.1 closeout merge has not been approved or performed**: `2810a56` is not an ancestor of `origin/main` (`6cbd44b`), and no approval is recorded in the repository or the instructions I received. I merged nothing. The AGENTS.md section 10 rule stands: the founder approves that specific merge, and the RTL chunk (`test_hardware_gate`, ~29 min) remains either to be run or waived. This design branch contains the closeout commits, so it should not be merged ahead of the closeout.
- No code, locked result, simulator, model or RTL file was modified. Deliverables are three documents plus the status line in `CURRENT_WORK.md`.

## What was produced

| deliverable | location |
|---|---|
| technical design (state, transitions, observations, rewards, deadline, terminals; twins; controllers; test plan; protocol; criteria; smallest implementation) | `docs/experiments/EXP-0013-phase1-2-design.md` sections 2-9 |
| baseline information-access matrix | same, section 5 |
| simulator test plan | same, section 6 |
| evaluation protocol and preregistered metrics | same, section 7 |
| literature comparison | `docs/experiments/EXP-0013-phase1-2-literature.md` |
| this decision memo | here |

## The design in five lines

1. v2-B: ordinary motion into a switch disc starts a hidden 12-step countdown, after which the gate in the divider is passable for exactly 8 steps; no observation shows the timer, and the gate is visible only on valid frames.
2. v2-A: same scene, gate on an exogenous periodic schedule with a deadline loose enough that waiting at the gate suffices; a control where map+memory must stay competitive.
3. Twins have bit-identical observations and image-channel-equivalent histories, differ only in the time since the agent entered the switch, and require different optimal first decisions (regret >= 3 steps).
4. Controllers: oracle; map+memory lifted from v1 (C1); hand-coded timer FSM (C2); stateless value policy (C3); probe-and-retry using bump feedback (C4, deliberately strong so the benchmark cannot win only against a strawman); learned arms C5/C6 are specified but not run.
5. Fresh locked held-out ids 3000-3999, 400 episodes with 400 distinct layouts per cell.

## Falsification gate (preregistered, section 8 of the design)

Validity gates V1-V6 on TRAIN/DEV2 (oracle solvable; non-predictive controllers lose to C2 by >= 6 steps median regret and >= 0.10 success in B; C2 near oracle; A is a true control; twins valid; pairing and tests pass). One TRAIN-only recalibration allowed; a second means stop. Then a single held-out run: GO-1 needs the same regret and success margins with bootstrap lower bounds above 3 steps and above 0. Failing any gate is a NO-GO with the negative result recorded, not a prompt to re-tune.

## What a GO would and would not mean

- It would show: **action-history predictive information matters on this task family** and is sufficient for a bounded observer (E1, E2).
- It would not show: that a learned model, a JEPA objective, ternary state or encoder skipping is needed or beneficial (E3-E6). The hand-coded C2 will probably solve v2-B, and v2-B cannot show a learned model beating a hand-coded predictor. A harder variant (per-episode latent timing, several interacting switches, stochastic delays) would be required for E5 and is deliberately out of scope.
- Honest prior (HYPOTHESIZED, not a prediction): V2 passes if the delay and window are tight relative to blackouts and the deadline rule leaves little slack; the real risk is that C4 (probe-and-retry) is competitive, in which case the result is a NO-GO that saves the cost of training.

## Recommendation

**GO for Stage 1 only: implement the v2 simulator, oracle and hand-coded controllers C1-C4 (software, no training, no GPU, about 3-4 engineering days ESTIMATE), with the abort points in design section 9, and evaluate the preregistered gates.**

**NO-GO, for now, on any neural training, learned world model, GPU sweep, or encoder/latency experiment for v2.** Those become eligible only after GO-1 passes on the locked held-out set and the founder authorizes them.

## What I need from you

1. Approve (or amend) the criteria in design section 8 *before* implementation, since they are meant to be fixed in advance. In particular: thresholds (6 steps, 0.10, 0.05), `DELTA=12`, `W=8`, deadline multiplier 1.25, and the single-recalibration rule.
2. Authorize Stage 1 as the one active objective.
3. Decide on the pending closeout merge (and the RTL chunk) so v2 branches from `main`.
4. Confirm the held-out ids 3000-3999 and that they may be added to the reserved-split guard.

STOP for founder review.
