# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Objective ID:** FT-0010 — Stage A

**Title:** Continuous edge-inference baseline integrity (EXP-0010)

**Status:** Stage A implemented and measured 2026-11-04; **awaiting founder + ChatGPT review. Stage B has NOT been started.** Outcome: the measurement/replay pipeline is reproducible and its two runtimes agree exactly, but the pretrained detector scores poorly under the frozen IoU 0.5 metric on the selected synthetic clips (holdout AP50 0.13; 0.34 at IoU 0.1), with mixed causes (box-extent mismatch with visible-extent labels, and class confusion with forklifts), so it is not yet a validated control for temporal-execution comparisons. See EXP-0010.

**Owner / agent:** Claude (implementation); founder + ChatGPT (research direction/review).

**Started:** 2026-11-04

## Why this objective

The founder has elected to investigate industrial edge inference as a *new exploratory commercial/technical branch*. Existing FlexTern RTL generic synthesis and 1B–3B ternary-LM memory analysis do not validate camera-inference power, quality or product economics. The earlier EXP-0007 datacenter negative result and EXP-0008/0009/0009A SRAM leakage/retention sensitivities remain intact.

This is a baseline-reliability task, not an attempt to prove ternary wins. Read experiments/EXP-0010-edge-inference-baseline.md for the preregistered method, controls, caveats and source references.

## One question

Can another engineer reproduce a **correct, optimized full-frame continuous-video inference baseline**—with audited data and model class coverage, exact stream timing, end-to-end latency, evaluation and run manifests—before FlexTern claims a new technique improves it?

## Scope — Stage A only

1. Audit AI City 2026 Track 1 single-camera RGB clips, per-frame 2D ground truth, dataset rights, scene splits, frame timestamps and class vocabulary. Synthetic warehouse footage is a proxy, not a factory deployment. The 2021 AI City set has a different vehicle-counting target; keep separate.
2. Select a person-detection task supported by the chosen checkpoint. Start with Torchvision SSDLite320-MobileNetV3-Large COCO if export and label compatibility can be proven. Record any necessary justified substitute.
3. Implement streaming replay at native temporal rate with the *same* decode/preprocess/postprocess pipeline in PyTorch FP32 reference and optimized ONNX Runtime CPU. Log every frame including late/dropped outputs. Ensure numerical/parity checks happen before performance claims.
4. Evaluate real ground-truth person-detection quality (AP50, recall/F1), p50/p95/p99 end-to-end latency, memory and stage timing; record complete dataset/model/environment hashes and exact commands.
5. Repeat on at least two disjoint scenes to show reproducibility and variance. Report explicitly what hardware, physical energy, customer constraints and model classes are **not** covered.

## Explicitly out of scope

Ternary or INT4 training, FPGA/ASIC design, a new Y-Array architecture, activation-cache optimization, claiming power savings without an external meter, large multi-week experiment backlogs, industrial safety certification, overwriting prior LM results, and declaring a business/product from synthetic data.

## Definition of done

- A runnable, documented, archived baseline pipeline producing identical semantics for the reference and deployment runtimes;
- rights/ground truth/model-label audit and scene-disjoint data manifest;
- at least two disjoint camera scenes evaluated end to end at source frame rate;
- raw predictions/traces, versioned reproducible configuration, parity/quality checks and repeat-run variance;
- an EXP-0010 results section distinguishing MEASURED from INFERRED and unmeasured;
- a review of whether event-driven experiments are justified, followed by founder decision **before** starting Stage B.

No speedup or energy advantage is required to complete Stage A.

## Required validation

Run and passing: `python -m unittest tests.test_edge_inference` (42 tests; 31 on host Python, all 42 in the WSL venv; 224 OK in one process together with the 182-test non-RTL baseline); parity, determinism, alignment, 7 offline runs and 14 real-time replays are in `results/edge_inference/` with manifests and per-frame traces. Existing RTL and non-RTL suites are unchanged. Commands: `analysis/edge_inference/README.md`.

## Stage A status against the definition of done

- Runnable, archived baseline with identical semantics for both runtimes — **done** (bit-identical offline repeats; torch-vs-ORT head diff 4e-5; AP50 equal).
- Rights / ground-truth / label audit and layout-disjoint manifest — **done** (CC-BY-4.0; 12 layout groups; splits by layout, not scene name).
- At least two disjoint scenes evaluated at source frame rate with repeats — **done** (development L09 and holdout L11, 3 B1 repeats on battery and 3 on AC each; calibration L10 offline only).
- Raw traces, versioned configuration, parity/quality checks, repeat variance — **done**.
- EXP-0010 results section separating MEASURED / INFERRED / unmeasured — **done**.
- Review of whether event-driven experiments are justified, then founder decision — **pending (the founder's step).** Evidence summary: pipeline trustworthy for CPU latency baselines; control quality not yet established (metric semantics and detector adequacy entangled); energy unmeasured. The single next experiment recommended in the handoff is a control-adequacy gate, not Stage B.

## Known uncertainties

Resolved by Stage A: hardware (one laptop CPU under WSL2), dataset availability/format/labels, ONNX export and postprocessing parity. Still open: real-world industrial-camera data access; customer-relevant deadline and safety requirements; physical power measurement (none exists); a detector that is adequate on this data; the cause of rare 100–840 ms stalls; AC/battery/thermal sensitivity of timing.

## Primary records

- docs/experiments/EXP-0010-edge-inference-baseline.md — protocol and later evidence.
- docs/PROJECT_STATE.md — latest verified technical capabilities through EXP-0009A.
