# MEMO 2026-11-07: Phase 1.2 adversarial decision memo (design Revision 3)

Purpose: argue against proceeding, as strongly as the evidence allows, then restate one recommendation. Written after the correction pass and the independent review; supersedes the recommendation paragraph of `MEMO-2026-11-07-exp-0013-phase1-2-decision.md` where they differ. Nothing has been implemented or run.

## The strongest objection

**The only difference between the hand-coded predictor (C2) and the strongest forecast-free controller (PTW) is one permitted counter and two constants.** PTW may count steps since it arrived at the gate and carries one tuned patience `Pi`; C2 may count steps since it pressed and knows `DELTA` and `W`. A GO therefore means: *a controller that is allowed to know the gate's timing constants plans at least 6 steps better than one that is forbidden to use "time since press" but may tune a patience on the same training layouts.* That outcome is largely decided by where we drew the information boundary, not by anything about prediction in the world. Worse, `Pi` tuned on TRAIN implicitly learns `DELTA`, so PTW partly *is* a model, disclosed but not removable. If PTW is tuned well the gap shrinks toward the scheduling benefit of overlapping the delay with picking; if PTW is restricted a little more the gap grows. The founder should judge whether a result so sensitive to one allowed counter is worth three to four days, and should read a GO as "the thesis was not ruled out by a designed puzzle", never as support for a world model.

## Other objections, with what the review found

1. **The twins are shakier than they looked.** Revision 2 contained two invalid twins (alpha was harmless; beta had no decision difference, `rho = 0`). The corrected beta and gamma exist only in a 1-D abstraction, need the switch-to-gate distance to be at least 5 moves (none exist at 3, marginal at 4), and need 3 to 5 further dark frames. Alpha fails the first-decision criterion at every distance. Until the real-simulator replay (test 7) runs, **no twin is validated**, and the layout generator may yield few.
2. **Circularity.** The delay, window and deadline are designer choices; the unchanged 6-step margin is also a designer choice. Mitigations (inherited multiplier, regret as the deciding metric, one recalibration, control variant A, strong PTW) reduce but do not remove the risk that the benchmark is tuned to its own answer.
3. **P3 relaxes a condition.** Replacing the 0.10 success margin by non-inferiority favours GO. It is justified by a ceiling effect, but it is a loosening and is offset only by the stronger baseline set and the SMALL-POSITIVE band.
4. **The task is tiny.** Deterministic, known dynamics, one switch, one gate: a 30-line state machine solves it. It cannot show that learning, JEPA, ternary state or encoder skipping matters, nor that adaptive perception has anything to decide (no frame is more informative than another). Its value is only as a *necessary* condition.
5. **Opportunity cost.** The informative experiment for FlexTern's thesis (uncertain consequences, selectively informative observations) is B2, which is sketched but not specified. Spending Stage 1 on v2-B first delays it.
6. **Boundary fragility.** The 80-step cap truncates `T_D` at 75 and may bind; R1-R4 are untested numbers on TRAIN. A cap collision costs the single recalibration before a result exists.
7. **Process risk.** The criteria were revised four times in two days, each time by me, with an advisor; the founder has not yet fixed them.

## What the objections do not undo

- Every failure mode above is a **cheap, early, pre-registered abort**: the first three engineering steps (simulator and tests, oracle and calibration, twin builder) run before any controller comparison and before any held-out data are touched, and they cost about two to three days of software with no compute.
- The infrastructure Stage 1 builds (episode-keyed gate dynamics, `O_h`/`O_c` access wrappers, fork/restore, twin builder, shared controller pipeline, held-out guard) is exactly what B2 would need, so the work is not wasted if v2-B is later judged too small.
- A NO-GO here is informative and cheap: it would say that even a task designed around hidden timing does not separate a model-based controller from the best forecast-free heuristic on this simulator family.

## What would change this recommendation

- If the founder's actual interest is the perception-cost and learned-model question, **specify B2 now** and treat v2-B only as its component test; Stage 1 should then be scoped as "infrastructure plus v2-B check".
- If the real-simulator replay yields no twins with `rho >= 3` at G = 5-7, stop (NO-GO on the benchmark family).
- If on TRAIN (where PTW is tuned and therefore favoured) the median regret gap between PTW and C2 is below 2 steps, stop before DEV2. This is a pre-registered abort, not a tuning opportunity.

## Recommendation (restated after the Rev 3 corrections)

**GO for Stage 1 as scoped in design section 10, with two additional pre-registered early aborts (no twins validated in the real simulator; TRAIN gap below 2 steps), and NO-GO on any training, learned world model, GPU sweep or latency experiment.** Confidence is moderate: the cost is low and every failure is early, but the founder should decide, before authorizing it, whether proving a prerequisite timing claim is worth building before B2. Approval of design section 8.2 (P1-P7) is required first; without it, the Rev 2 criteria stand.

STOP for founder review.
