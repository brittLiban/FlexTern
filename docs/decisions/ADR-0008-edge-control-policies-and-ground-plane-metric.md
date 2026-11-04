# ADR-0008 — Edge-video control: two monitoring policies and a ground-plane zone metric

Status: Proposed (agent proposal; the founder decides acceptance; effective only if accepted)
Date: 2026-11-04

## Context

EXP-0010B's image-grid zone metric failed for reasons that were mostly metric construction (humanoid robots, frame-edge truncation, one-sided boundary tolerance) and left open what the application actually is. EXP-0010C attributed the residual errors, measured localization in metres, surveyed class mix, and evaluated a corrected metric under two policies on two fresh layouts (`docs/experiments/EXP-0010C-control-semantics-validation.md`).

## Decision (proposed)

1. **Two applications are evaluated separately and never merged.** A, human-only monitoring: humanoid robots are negatives. B, person-shaped-agent occupancy: Person and humanoid robots are positives. Vehicles, AMRs and transporters are negatives under both. Humanoid semantics are a customer decision; no policy ignores humanoids to raise a score.
2. **The primary metric is ground-plane zone occupancy**: 3 m floor cells, positions from the 3D location (GT) and the back-projected box bottom-centre (detections), tolerance 0.75 m (the measured 95th-percentile localization error), NEG = no positive agent within 2 x tolerance, ignored/truncated agents and truncated detections handled symmetrically, and gating on TPR and false alarms per predicted cell (FPR and balanced accuracy reported only; they depend on the number of empty cells).
3. The 3x3 image grid (ADR-0007 point 1 as implemented in EXP-0010B) is demoted to a comparison metric.
4. Every pass/fail bar is labelled TARGET until a customer requirement exists, and an outcome-to-recommendation decision rule is fixed before any fresh holdout.
5. Fresh-holdout runs require a committed lock, a clean `analysis/` and `tests/`, and manifests that record the whole-repository state.
6. Operating thresholds must be chosen with a criterion that does not reward recall at the cost of precision (the EXP-0010C balanced-accuracy rule chose the lowest threshold and was flawed).

## Alternatives Considered

Keep the image grid (physical meaning varies by orders of magnitude, R3); ignore humanoids globally (would hide the human/robot ambiguity that decides application A); a single merged policy (hides which application a result serves).

## Why

Localization error is uniform in metres, not pixels (EXP-0010C R2); class semantics decide the outcome on nearly all unused layouts (R4); the metric change alone flipped one verdict (R7), so policies and parameters must be locked before holdout data.

## Consequences

Later quality claims state the policy and use this metric. ADR-0007 points 3 and 4 are carried forward in point 1 and 5 here. The EXP-0010C result (policy B passes one layout and fails one; policy A fails both) does not establish a control.

## Evidence

EXP-0010C R1-R9. Agent proposal; policy semantics and the bar are founder decisions.

## Supersedes

ADR-0007 (Proposed), if this ADR is accepted.

## Superseded By

None.
