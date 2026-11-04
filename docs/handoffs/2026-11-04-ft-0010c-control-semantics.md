# Handoff

Date: 2026-11-04
Agent: Claude (Sonnet 5.5), advisor review by Opus 5.5 (plan and completion)
Current commit: see `git log` (lock commit 97cbd9b; fresh runs executed from it with analysis/ and tests/ clean)

## Objective

FT-0010 EXP-0010C Control Semantics and Residual Error Validation. Stage B was NOT started.

## What I Completed

- Attributed EXP-0010B's residual false alarms by triggering detection (humanoid 62-70%, frame-edge truncation 11-25% on L08/L02; nothing unexplained), measured ground localization (0.2-0.4 m median, 0.5-0.8 m p95), surveyed class mix (humanoids in nearly every frame of every unused layout) and agent-level misses (occlusion, small people).
- Defined two applications (A human-only, B person-shaped-agent) and a ground-plane zone metric with measurement-derived parameters; locked protocol_v3 + taus + fresh clips + a decision rule before any fresh video was fetched; scored YOLOX-Tiny once per fresh clip (L05, L03) with the EXP-0010B v2 metric as written beside it; post-hoc checks; a short timing comparison (on battery).
- Result: preregistered decision CHANGE_WORKLOAD (B: L05 met the bar, L03 narrowly missed on precision; A failed both). Recommendation to the founder: change the application/workload (the change must come from a founder-set requirement, not from what this detector passes); do not start temporal execution; do not keep tuning this detector or metric.

## What I Changed

New: `analysis/control_semantics/`, `tests/test_control_semantics.py`, `results/control_semantics/`, `analysis/control_gate/posthoc/` (the EXP-0010B scratch scripts committed byte-for-byte), EXP-0010C record, ADR-0008 (Proposed), this handoff, the review. Edited docs: `CURRENT_WORK`, `PROJECT_STATE`, `KNOWN_ISSUES` (KI-015 mitigated, KI-018, KI-019), `OPEN_QUESTIONS` (Q-025/Q-027 narrowed, Q-028), ADR-0007 pointer. No EXP-0010B or Stage A result, protocol or code was modified.

## Tests / Validation Run

`python -m unittest tests.test_control_semantics tests.test_control_gate tests.test_edge_inference`: 78 tests OK on host Python (11 skipped: need av/cv2/torch/ijson); the new module's tests are numpy-only. The 182-test non-RTL baseline was not rerun (not affected: no existing code changed).

## Actual Results

See EXP-0010C R1-R9: fresh B L05 TPR 0.992 / false alarms 0.042 (bar met), L03 0.919 / 0.172 (not met); A fails both (0.57-0.79 false alarms); v2 as written 0.715 / 0.883 balanced accuracy; no threshold meets the pair on all six clips; timing on battery YOLOX-Tiny 8 thr p50 49-73 ms, p99 88-109 ms. No energy measured.

## Remaining Problems

- Which application is the product, and what minimum person size/visibility must be served (Q-028); a model-side objective is justified only by those answers.
- The locked tau rule was flawed (KI-018); L05's pass rests on 17 cells and a minority of untruncated agents; the 3D-box ground truth is validated per scene but not independently.
- Timing was on battery (a deviation from protocol_v3, which specified AC); no energy; no optimized decode/postprocess; no INT8/FP16.
- Coverage: the v3 metric ignores truncated people (70% of B agents on L05), so passes validate the image interior only. The 0.75 m margin does not hold for AgilityDigit (R2b).
- The first false-alarm classifier lumped partial overlaps into `other_object`; the corrected v2 classifier and L03 crop audit are in EXP-0010C R8 (original files kept).
- Manifests made before the completion review show the wrong `command` field (see EXP-0010C Limitations).

## Important Context

- Provenance: new manifests record the whole-repository state (commit, dirty paths, tree hash); git is pinned with `-c core.autocrlf=true` because WSL otherwise sees all files as modified. EXP-0010B manifests are unchanged and still say b056ace/clean.
- Fresh layouts L05 and L03 are spent; L06 and L07 remain unused in reserve; L01 and L04 had no eligible cameras.
- Out-of-repo side effects: WSL cache additions (two fresh videos ~240 MB, new detection caches under `cag/dets/hold_L05|hold_L03`); nothing else installed; no Windows settings, power plans or apps changed; the laptop was on battery for the timing runs and drained to about 77%. No background jobs were reaped this time (all long jobs ran in the foreground, one process at a time).
