# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Objective ID:** FT-0010 — EXP-0010C Control Semantics and Residual Error Validation

**Title:** Can a trustworthy industrial-vision quality control be established without repeatedly changing models and evaluation criteria?

**Status:** In progress (started 2026-11-04, authorized by the founder after review of EXP-0010B). Stage B has NOT been started and will not be started by this objective. The previous objective (EXP-0010B Control Adequacy Gate) is complete: no adequate control among the tested configurations; its records are unchanged (`experiments/EXP-0010B-control-adequacy-gate.md`).

**Owner / agent:** Claude (implementation); founder + ChatGPT (research direction/review).

## Scope (from the founder's authorization)

1. Diagnose YOLOX-Tiny's residual false alarms on existing data with clearly labelled post-hoc diagnostics (detector error vs annotation problems vs humanoid ambiguity vs frame-edge truncation vs zone boundary).
2. Define one internally consistent application metric; distinguish human-only monitoring from general agent/robot occupancy; no automatic ignoring of humanoids to raise scores.
3. Decide whether the 3x3 image grid is representative or ground-plane regions are necessary.
4. Lock the corrected protocol, thresholds and pass/fail criteria before a completely fresh, layout-disjoint holdout.
5. YOLOX-Tiny remains the principal baseline; no model or training change unless the diagnosed failure mode justifies it (and then as a separate recorded decision).
6. Record the complete git state in every manifest; preserve all earlier results.
7. Tests, reproducible benchmarks; document unmeasured energy, thermal uncertainty and missing production optimization.
8. Update records, commit, obtain an independent review; **stop** and give an explicit recommendation: proceed to temporal execution, change the application/workload, or stop this branch.

## Primary records

`experiments/EXP-0010B-control-adequacy-gate.md`, `handoffs/2026-11-04-ft-0010-control-adequacy-gate.md`, `reviews/REV-2026-11-04-ft-0010-control-adequacy-gate-review.md`, `decisions/ADR-0007-edge-control-application-metric.md` (Proposed). This objective's record will be `experiments/EXP-0010C-control-semantics-validation.md`.
