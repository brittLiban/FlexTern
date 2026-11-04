# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Objective ID:** FT-0010 — Control Adequacy Gate (EXP-0010B)

**Title:** Establish a scientifically defensible detection control before any temporal or ternary work

**Status:** Completed 2026-11-04; **awaiting founder + ChatGPT review. No further objective is active. Stage B has NOT been started.** Outcome: **no adequate control among the tested configurations** under the preregistered zone-occupancy bar on two fresh, layout-disjoint holdouts (selected YOLOX-Tiny: balanced accuracy 0.70 / 0.68, TPR ≥ 0.98, FPR 0.59 / 0.61; the Stage A detector 0.71 / 0.62). Post-hoc, the metric itself inflated false alarms (humanoid robots excluded by the labels, about 40% of the cell-level FPR; people cut off at the frame edge; one-sided boundary tolerance; a gap in my metric and layout rule, EXP-0010B R6); with those corrected YOLOX-Tiny reaches balanced accuracy 0.84 / 0.86 at TPR >= 0.98 but FPR is still 0.31 / 0.25, far from 0.10. See `experiments/EXP-0010B-control-adequacy-gate.md`.

**Owner / agent:** Claude (implementation); founder + ChatGPT (research direction/review).

**Started / finished:** 2026-11-04

## Question answered

Is there a detector/input configuration that is an adequate quality control for fixed-camera person monitoring (application-level metric, fresh layouts, 30 fps on the CPU baseline)? For the tested configurations: **no**. The measurement/replay pipeline from Stage A is trustworthy for CPU latency baselines; the quality control is not yet trustworthy.

## What was established (details in EXP-0010B)

- Task distinctions and primary metric (zone occupancy; Proposed ADR-0007); quality bar labelled TARGET.
- 3D+calibration ground truth reproduces visible boxes only in some scenes (rule preregistered; L09 fails).
- Candidates S0/S1/S2 (SSDLite variants) and YOLOX-Tiny passed parity; YOLOX-Tiny is far better on boxes (AP50 0.62–0.79 vs 0.11–0.31 on fresh layouts) but needs 8 CPU threads for 30 fps.
- Timing on AC power with interleaved repeats; host clock/thermal state confounds documented. No energy was measured.
- Not done: INT8/FP16 or other optimized-pipeline comparators (deferred because the FP32 control failed), YOLOX-S, fine-tuning.

## Validation

`python -m unittest tests.test_control_gate` (18 tests, host Python) and `tests.test_edge_inference` (WSL venv) pass; Stage A artifacts and RTL unchanged. See the handoff for exact counts and commands.

## Decision needed from the founder (with ChatGPT)

1. Humanoid-robot semantics for the metric (ignore / person / separate; ADR-0007 proposes ignore).
2. Whether to spend one more objective on a corrected, humanoid-aware protocol with a **new** fresh holdout (the current holdout is spent), or to start temporal-execution experiments against YOLOX-Tiny with the metric caveats stated. My recommendation: not yet Stage B. Run one more corrected-gate objective (humanoid- and ignore-aware metric with a tolerant NEG band, scored on a new fresh holdout whose layout rule screens class mix); temporal experiments would otherwise measure retention against a control whose own false-alarm rate is 0.25-0.6.

## Primary records

- `experiments/EXP-0010B-control-adequacy-gate.md` (this objective), `experiments/EXP-0010-edge-inference-baseline.md` (Stage A, preserved).
- `PROJECT_STATE.md`, `handoffs/2026-11-04-ft-0010-control-adequacy-gate.md`, `reviews/REV-2026-11-04-ft-0010-control-adequacy-gate-review.md`, `decisions/ADR-0007-edge-control-application-metric.md` (Proposed).
