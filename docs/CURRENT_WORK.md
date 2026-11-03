# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Objective ID:** FT-0010 — Stage A

**Title:** Continuous edge-inference baseline integrity (EXP-0010)

**Status:** Active — protocol created 2026-11-04; implementation and measurements not yet performed.

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

No tests or benchmarks have been run for EXP-0010 as of this status change. Implementer must supply actual commands, output and raw logs; do not invent results. Existing RTL and non-RTL suites should remain unchanged.

## Known uncertainties

Available benchmark hardware; availability/format of selected dataset clips and 2D labels; ONNX export compatibility and postprocessing parity; real-world industrial-camera data access; customer-relevant inference deadline and safety requirements; future gross physical device power measurement.

## Primary records

- docs/experiments/EXP-0010-edge-inference-baseline.md — protocol and later evidence.
- docs/PROJECT_STATE.md — latest verified technical capabilities through EXP-0009A.
