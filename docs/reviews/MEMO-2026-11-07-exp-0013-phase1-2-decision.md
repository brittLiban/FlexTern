# MEMO 2026-11-07: Phase 1.2 decision (design and falsification gate), Revision 3

To: founder (for discussion with ChatGPT). From: implementation agent. Status: recommendation; nothing implemented. The adversarial counter-argument is in `MEMO-2026-11-07-exp-0013-phase1-2-adversarial.md`; read both. Earlier text of this memo is in git history (`5e042f0`).

## Repository and approval state

- Work is on `exp-0013-phase1-2-design`, created from `exp-0013-phase1-1-closeout` @ `2810a56`.
- **The Phase 1.1 closeout merge has not been approved or performed**: `2810a56` is not an ancestor of `origin/main` (`6cbd44b`) and no approval is recorded. I merged nothing. The branch carries the closeout commits, so it must not be merged ahead of the closeout. The RTL chunk (`test_hardware_gate`, ~29 min) is still unrun or unwaived.
- Only documentation was changed. A throwaway reference script (scratchpad, not committed) checked the twin arithmetic.

## Deliverables

| deliverable | location |
|---|---|
| technical design (state, scheduled vs realized gate, transitions, observations, reward, deadline and boundary audit, terminals) | `docs/experiments/EXP-0013-phase1-2-design.md` sections 2 |
| counterfactual twins (definitions, derivation, G-sweep, witnesses) | same, section 3 |
| controllers, oracle split, baseline information-access matrix | same, sections 4-5 |
| simulator test plan; evaluation protocol | same, sections 6-7 |
| falsification gates (Rev 2 registered vs Rev 3 proposed), prerequisite-only claim, B2 sketch | same, sections 8-9 |
| literature comparison | `docs/experiments/EXP-0013-phase1-2-literature.md` |
| reviews | `REV-2026-11-07-exp-0013-phase1-2-design-review.md`, `...-correction-review.md`, the adversarial memo |

## What the corrections changed in substance

1. **Two of the three original twins were invalid.** Revision 2's alpha (pending vs idle at the switch) was harmless; its beta had no decision difference (pooled regret 0). Corrected beta and gamma exist (pooled regret 6-7 in the abstraction) but only when the switch-to-gate distance is at least 5 moves (none at 3, marginal at 4), so placement is narrowed to 5-7 moves. Alpha fails the "different optimal first decisions" criterion at every distance and is demoted to a demonstration pair. **No twin is validated until it is replayed with the real oracle (Stage 1 test 7).**
2. **Stronger baseline:** press-travel-wait (PTW, with press-before-pick), same parser/planner/observation access as the timer tracker C2, joins the non-predictive set. Its tuned patience partly encodes the delay; the C2-PTW gap measures the value of the model, not proof that PTW holds none.
3. **Gate dynamics:** scheduled (machine / periodic) and realized (held-open) states separated, nine testable invariants.
4. **Oracles:** `O_h` defines regret; `O_c` is diagnostic only.
5. **Deadline:** inherited 80-step cap, `T_D` truncated at 75, per-episode feasibility requirements, `+(P - O)` rule for v2-A.
6. **Criteria:** the registered Rev 2 gate table is preserved row by row beside the Rev 3 proposals P0-P7. The regret margin (6 steps, lower bound > 3) is unchanged; the conjunctive success margin is replaced by non-inferiority (a loosening, disclosed); outcomes partition on the lower bound (GO / SMALL-POSITIVE / NO-GO), weaker of LA_NOM and LA_OCC.

## What a GO would and would not mean

- **Would show:** forecasting the gate from one's own action history beats the strongest forecast-free controller on this task family, and the information is sufficient for a bounded observer (E1b, E2).
- **Would not show:** that a learned model, JEPA objective, ternary state or encoder skipping helps (E3-E6). v2-B is deterministic with known dynamics; a 30-line state machine solves it; nothing in it makes perception selectively informative. It is a **prerequisite** (necessary-condition) test. B2 (uncertain consequences, selectively informative observations) is the benchmark where adaptive perception and learned state could matter; it is sketched, not specified.

## Recommendation

**GO for Stage 1 only**: implement the v2 simulator, `O_h`, twin builder and hand-coded controllers C1/C1+/PTW/C2/C3/C4 in software (no training, no GPU; about 3-4 engineering days, ESTIMATE), with the abort points in design section 10 plus two pre-registered early aborts (no twins validated in the real simulator with pooled regret >= 3; TRAIN median regret gap between PTW and C2 below 2 steps). **NO-GO, for now, on any neural training, learned world model, GPU sweep, or latency experiment**; they become eligible only after GO-1 on the locked held-out set and explicit authorization. Confidence is moderate; the adversarial memo explains why a GO is sensitive to where the information boundary between PTW and C2 is drawn, and why the founder may prefer to specify B2 first.

## What I need from you

1. Approve or amend the Revision 3 proposals P0-P7 (design 8.2): margin 6 steps with lower bound > 3, non-inferiority 0.02, the SMALL-POSITIVE band, weaker-of-two-conditions, `DELTA = 12`, `W = 8`, inherited 1.5x deadline truncated at 75, switch-to-gate distance 5-7 moves, twin yield minimums. Until you do, the Rev 2 registered criteria stand.
2. Decide whether Stage 1 should be v2-B-first or whether B2 should be specified first (adversarial memo).
3. Decide the order of the pending closeout merge and the RTL chunk.
4. Confirm held-out ids 3000-3999 and their addition to the reserved-split guard.

STOP for founder review.
