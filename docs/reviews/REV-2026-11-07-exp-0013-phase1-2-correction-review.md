# REV-2026-11-07: Phase 1.2 design, correction pass and independent review of Revision 3

Status: advice and self-audit, not an accepted decision. Two parts: what the correction pass changed, then the independent reviewer's points on the revised document and their dispositions.

## Part 1: correction pass (founder directive, items 1-8)

| # | directive | what was done |
|---|---|---|
| 1 | v2-A: exogenous schedule vs agent-dependent deferred closure | Scheduled/machine state `M_t` and realized state `g_t` separated (`g_t = open(M_t) OR (g_{t-1} AND occupied_t)`); exact step order; nine testable invariants I1-I9 with an induction proof of the no-trapping invariant. |
| 2 | re-derive the twins; correct beta; executable witnesses; mark unproven twins hypothetical | Throwaway exhaustive-search script (not committed) in a 1-D route abstraction. **Revision 2 beta was not a twin** (`rho = 0`: both states are best served by "go"); Revision 2 alpha was harmless. Valid beta/gamma pairs found, witness scripts built and replayed (identical observation, different hidden state, image-equivalent histories). Real-simulator status is HYPOTHETICAL until test 7. |
| 3 | strong press-travel-wait baseline, press-before-pick allowed | PTW added (press-after-pick, press-before-pick, non-forecasting rule variant), same parser/planner/observation access as C2 via a pluggable gate policy; added to the non-predictive set. |
| 4 | oracle separation | `O_h` (current hidden state; defines regret) and `O_c` (clairvoyant; diagnostic only) with access tables and invariance tests. In v2 as specified their physical finish times coincide (blackouts and noise do not act on physics); they differ for perception metrics and for stochastic variants. |
| 5 | deadline and boundary audit | Inherited 80-step cap kept (`T_CAP = 80`, no override); the Rev 2 `T_CAP = 2 x p95` withdrawn because it could exceed 80; `T_D` truncated at 75; per-episode requirements R1-R4 and boundary tests E1-E10. |
| 6 | prerequisite-only claim; B2 | Section 9 states what v2-B does and does not test; B2 sketched (uncertain consequences, selectively informative observations), not specified. |
| 7 | thresholds after adding the stronger baseline | Proposals P0-P7 with reasons; registered Rev 2 criteria kept as the operative set until approved; history preserved in the appendix and in 8.1. |
| 8 | independent review and adversarial memo | Part 2 below and `MEMO-2026-11-07-exp-0013-phase1-2-adversarial.md`. |

## Part 2: independent reviewer's points and dispositions

| # | point | disposition |
|---|---|---|
| 1 (blocking) | The rewrite deleted the preregistration: Section 8.1 became a summary and later text cited gates no longer defined. | Restored as one full gate table with a Rev 2 column and a Rev 3 column, row by row (V1-V6, GO-1, recalibration). |
| 2 (blocking) | Alpha fails the founder's "different optimal first decisions" criterion; the rewrite quietly relaxed it by making pooled regret the test. | Criterion kept (T4a disjoint first actions) and pooled regret added (T4b). Alpha reclassified as **not a twin** and removed from the V5 yield; it remains a demonstration pair for event memory. A G-sensitivity sweep confirmed alpha never satisfies T4a at G = 3-7. |
| 3 (blocking) | Outcome rules left a gap (estimate >= 6 with lower bound in (0, 3] was neither GO nor SMALL-POSITIVE). | Partition on the lower bound: GO (`est >= 6` and `LB > 3`, plus P3 and A-control), SMALL-POSITIVE (`LB > 0`, not GO), NO-GO (`LB <= 0`). Overall outcome is the weaker of LA_NOM and LA_OCC (P7). |
| 4 (blocking) | R3 (`T_D,A >= p95_A + P`) fails on the document's own numbers (~79 > 75), consuming the recalibration before B is calibrated. | R3 changed to `+ (P - O)`: the worst extra wait at a closed gate is the closed part of the period (12), giving ~67 <= 75 with margin. Chosen from cap arithmetic, before any result. |
| 5 | Run the G-sensitivity check before claiming twins exist. | Done (design 3.4). **No beta/gamma twins at G <= 3, marginal at G = 4, robust from G = 5**; placement narrowed to G = 5-7 (geodesic 0.25-0.35). |
| 6 | Regret SD "at most 15" is unjustified (censoring at 80 gives a ~50-step range); update memo, Q-029 and CURRENT_WORK to Rev 3 values. | SD assumption corrected to ~25 (labelled an assumption); cap-determined medians flagged with a mean-regret fallback report; memo, Q-029 and CURRENT_WORK updated. |
| 7 | Persist this review, write the adversarial memo leading with the C2/PTW information-restriction objection, restate the recommendation, include the co-author line, push the branch. | Done: this file; the adversarial memo leads with exactly that objection and adds two early aborts; the final commit carries the attribution line and the branch is pushed (never `main`). |

Effect on the recommendation: points 2-5 tightened the criteria and the twin claims; the substance (Stage 1 software only, no training) is unchanged, with two extra pre-registered early aborts.
