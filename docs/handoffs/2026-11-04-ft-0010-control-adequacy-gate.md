# Handoff

Date: 2026-11-04
Agent: Claude (Sonnet 5.5), advisor review by Opus 5.5
Current commit: recorded in the final commit message and in `git log` (the experiment record points here)

## Objective

FT-0010 Control Adequacy Gate (EXP-0010B). Stage B was NOT started.

## What I Completed

- Defined and preregistered (`analysis/control_gate/protocol_v2.json`) the task distinctions (detection / presence / count / zone occupancy), the primary metric (zone-occupancy balanced accuracy), a TARGET quality bar, candidate set, real-time feasibility rule, selection rule, holdout layout rule and failure outcome. Stage A artifacts and `protocol.json` are untouched.
- Validated 3D+calibration ground truth (passes on L10/L11 and both fresh layouts, fails on L09).
- Built candidates S0/S1/S2 (SSDLite 320 squash, 576x320, 416x240) and YOLOX-Tiny, with parity checks; diagnosed S0 errors (box extent vs occlusion, aspect, vehicle confusion with a 30-crop audit).
- Selected YOLOX-Tiny (8 threads) by the preregistered rule; chose and scored two fresh layouts once each (L08, L02); ran labelled post-hoc diagnostics; ran interleaved 30 fps timing replays on AC power.

## What I Changed

New: `analysis/control_gate/`, `tests/test_control_gate.py`, `results/control_gate/`, `docs/experiments/EXP-0010B-control-adequacy-gate.md`, `docs/decisions/ADR-0007-edge-control-application-metric.md` (Proposed), this handoff, the review file. Edited docs: `CURRENT_WORK`, `PROJECT_STATE`, `KNOWN_ISSUES` (KI-015..017), `OPEN_QUESTIONS` (Q-025 narrowed, Q-027), a pointer in `EXP-0010`. No RTL, prior analysis code or earlier results touched.

## Tests / Validation Run

`python -m unittest tests.test_control_gate tests.test_edge_inference`: 60 tests OK on host Python (11 skipped: need av/cv2/torch/ijson) and 60 OK in the WSL venv (nothing skipped). A wider non-RTL run is described in the final report if it completed.

## Actual Results

See EXP-0010B. Short form: no tested configuration meets the preregistered bar on the fresh holdouts. YOLOX-Tiny: balanced accuracy 0.701 (L08) / 0.682 (L02), TPR ≥ 0.98, FPR 0.59 / 0.61; Stage A SSDLite320: 0.705 / 0.616. Post-hoc: ~80% of the confident false alarms are humanoid robots excluded by the labels; excluding those zones YOLOX-Tiny reaches 0.828 / 0.794, FPR 0.34 / 0.39. YOLOX-Tiny needs 8 threads for 30 fps (p50 29–38 ms; 4 threads drops 12–20%). No energy was measured.

## Remaining Problems

- Methodology gap on my side: zone metric ignored the humanoid ignore-region policy and the layout rule did not screen class mix (KI-015). The holdout is spent for this protocol.
- Ground-truth foot-point noise (KI-016) and L09 failing geometry.
- Timing confounded by host clock/thermal state; one repeat at the real-time bar's edge (p99 99.4 ms).
- YOLOX-S not evaluated; INT8/FP16 and a faster decode/postprocess path not done.

## Important Context

- Founder + ChatGPT decisions needed: humanoid semantics; whether to run one more objective (corrected humanoid-aware protocol, new fresh holdout) before Stage B.
- The detection caches, videos, ground truth and ONNX files live outside git in WSL `~/flextern_edge` (new: `cag/`, `models/yolox/`, SSD variant ONNX/anchors, 8 more scenes' ground truth ~1.4 GB, two holdout videos). The holdout videos and GT were inspected only through post-hoc diagnostics after scoring.
- Env gotchas as in the Stage A handoff. Shell note: heredocs containing apostrophes broke the Bash tool, so scripts were written to files. Windows memory protection killed two background jobs; run jobs one process at a time (KI-017).
- Out-of-repo side effects: WSL cache additions above; downloaded YOLOX-Tiny ONNX (20 MB), a partial YOLOX-S ONNX (about 11 MB of 35.9 MB, unusable), Megvii reference scripts (`demo_utils_ref.py`, `onnx_inference_ref.py`, read before use); a scratch directory under the Claude temp dir with helper scripts; no Windows settings, power plans or apps were changed; the laptop was on AC power for the timing runs.
