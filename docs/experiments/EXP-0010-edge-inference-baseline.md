# EXP-0010 — Continuous edge-inference baseline (FT-0010)

Status: Not started — experiment specification approved for initial baseline work; no measurements
Date: 2026-11-04
Commit: None for measured results; this document establishes protocol only

## Research Question

Can any FlexTern execution strategy measurably reduce **gross end-to-end system energy per camera-hour** on continuous, fixed-camera visual inference relative to an appropriately optimized, task-equivalent conventional baseline, without unacceptable losses in object detection, event recall or response latency?

**Current objective is only baseline integrity (Stage A), not proving the full thesis.**

## Hypothesis

- **HYPOTHESIZED:** Continuous fixed-camera video may have exploitable temporal redundancy.
- **HYPOTHESIZED:** A useful temporal-execution method could avoid expensive inference while retaining event recall.
- **TARGET, not forecast:** Ultimately test a >=2.0x gross-energy advantage over the strongest practical conventional competitor, across distinct workloads and two hardware classes, subject to quality and latency constraints.
- **Uncommitted:** Ternary arithmetic is a later controlled variable. It is not assumed to help object detection, and no existing ternary model is silently substituted for a trained/evaluated detector.

## Why It Matters

The current FlexTern thesis, RTL measurements and SRAM/LPDDR analyses concern mostly ternary-language-model decode. They do **not** demonstrate physical power, cost or accuracy advantages for continuous camera inference. EXP-0007 showed batching can destroy a resident datacenter advantage; EXP-0008/0009/0009A showed idle leakage, temperature, retention implementation and backing store can change an edge-memory verdict. EXP-0010 begins a **new, exploratory workload branch**, not an established extension of the old processor to computer vision.

The result may support a software runtime, specialized accelerator, different market, or stopping this branch. Negative results are publishable project evidence.

## Stage A — Active Objective (only)

Build a reproducible streaming-video *baseline* and validity audit before testing FlexTern enhancements.

1. **Dataset selection / rights audit.** Primary candidate: AI City Challenge 2026 Track 1 Physical AI Smart Spaces, single RGB fixed-camera views with per-frame 2D labels; use a selected subset of train/validation scenes and keep scenes disjoint across calibration/development/final holdout. Confirm actual files, class coverage, rights, annotation format, timestamps, and train/validation split before training or scoring. The corpus is synthetic: DO NOT call the result a real-factory validation.
2. **Task restriction.** Initial task is **person detection/occupancy on one camera** where a compatible pretrained detector can supply a legitimate baseline. Forklifts, pallet trucks and robot classes require separately demonstrated model coverage or fine-tuning. Multicamera 3D-HOTA is the challenge's original task, not our Stage A metric; report that benchmark mismatch explicitly.
3. **Reference model.** Start with Torchvision SSDLite320-MobileNetV3-Large COCO weights for transparency. If ONNX export or compatible model postprocessing causes difficulties, record the blocker and use a documented, fixed alternative rather than silently changing methods. A second small detector (e.g. YOLOv8n) is a later cross-model test, with model license review.
4. **Execution.** Run on available hardware, first PyTorch FP32 correctness reference and ONNX Runtime CPU full-inference baseline, streaming every frame at the *source* FPS with identical decode, resize, preprocessing, NMS, threshold and output schema. Use fixed CPU threads, process priority, power mode, and explicit warmup. Do not require a GPU or purchase hardware for Stage A.
5. **Scoring.** Use verified ground-truth 2D boxes and a fixed IoU/confidence policy for person AP50, F1 and recall; report unmatched/unknown frames instead of inventing labels. Event/episode recall may be reported only after ground-truth entry/exit episodes and acceptable detection deadline are explicitly defined and audited. A model whose class vocabulary cannot detect the chosen class is not a valid quality control.
6. **Logging.** For each frame record source timestamp, arrival/processing/output timestamps, decode/preprocess/inference/postprocess time, predictions, dropped frames and decision mode. For each run record dataset/model SHA-256, Git commit, environment (OS, Python, PyTorch, ONNX Runtime, OpenCV), EP, threads, thermal/power settings, repeats and run duration.
7. **Energy boundary.** Gross J/camera-hour is a later physical metric requiring external power measurement at a documented device boundary while **replaying in real time**, including decode and idle. Early telemetry (CPU utilization, CPU/GPU vendor counters, latency) is *diagnostic only*; no estimated energy, PyTorch speed, FLOP reduction or Yosys cell count may be presented as measured device energy.
8. **Baseline validation.** At least two disjoint camera scenes and repeated runs; model parity checked before timing; report the magnitude of repeat-run variance. Predeclare accuracy tolerance and application response deadline before subsequent optimization sweeps. Once a final holdout is locked, do not tune on it.

**Stage A definition of done:** another engineer can retrieve the documented dataset subset, inspect the environment/manifest, run a full-every-frame reference and one optimized CPU inference implementation, reproduce per-frame outputs and timing, and see an explicit statement of all metrics **not yet measured**. No gain claim is necessary.

## Later comparisons (not current commitments)

After the founder reviews Stage A, possible next objectives are conventional MOG2 gating; detector every N=2/4/8 frames plus tracking; a validated event scheduler; oracle temporal bound; activation caching; hardware-supported FP16/INT8 with representative calibration; only then a properly trained/fine-tuned ternary control. One objective is selected at a time, consistent with AGENTS.md.

Important controlled comparisons:
- Temporal approach vs **best conventional gating/tracking** at equal downstream event semantics.
- Ternary + temporal vs **INT8 + identical temporal strategy**, not ternary vs naïve FP32.
- Gross physical energy, not only inference latency or skipped MACs.
- Paired camera-level trials, not independently sampled frames from the same clip.

## Baseline

B0: full-frame PyTorch FP32 reference (quality/correctness only).
B1: full-frame ONNX Runtime CPU, same model and end-to-end pipeline; optimized deployment CPU control.
Later B2: supported INT8 with actual calibration/accuracy report.
Later B3: optimized hardware-specific EP/native baseline (TensorRT/OpenVINO/QNN as supported); no claims when hardware is absent.
Later B4/B5: motion gate and keyframe+tracker.

## Method

- Verify dataset files, classes, GT, frame rates, and license; create immutable subset manifest and train/validation/holdout assignment **by scene**.
- Freeze detector weights, preprocessing, input resolution, confidence and IoU thresholds; compute checkpoint hash.
- Establish a single canonical decoded frame, prediction and trace schema. Calibrate output parity and ensure model/framework postprocessing semantics match.
- Replay the same encoded video at original timestamps. Record how dropped/late frames are scored instead of silently omitting them.
- Record CPU timings for all stages, host memory and system utilization. Do not infer energy from these metrics.
- Repeat and archive raw per-frame outputs, run manifests, and a small readable summary.

## Metrics

Stage A: AP50, F1, recall, dropped/deadline-missed outputs, p50/p95/p99 arrival-to-output latency, per-stage wall time, host memory and pipeline FPS; identify whether every ground-truth frame and event was scored.

Later physical: gross J/camera-hour (integrate measured DUT input power over real replay time), Wh/camera-day, event miss and false alarm rates per camera-hour, thermal stability, cost/camera-month. Report idle-subtracted energy only as supplementary diagnostics.

## Results

**NOT MEASURED.** No video benchmark, device power result, accuracy result, or FlexTern improvement has been produced under EXP-0010.

## Interpretation

**NOT YET POSSIBLE.** A future gain over an unoptimized PyTorch model would not establish differentiation from commodity runtimes. Static or synthetic results cannot alone justify real-industrial safety claims or custom silicon.

## Surprises

None recorded.

## Limitations

Dataset access and exact annotation compatibility not yet audited; hardware available to the implementing agent not recorded; detector class coverage is restricted; industrial customer requirements not yet measured; synthetic warehouse validation cannot replace a customer-owned stream. Physical meter and cross-hardware comparisons are deferred.

## What This Proves

Nothing yet. It states falsifiable methods and prevents benchmark inflation.

## What This Does NOT Prove

No validated edge video speedup, physical energy savings, camera BOM reduction, product-market fit, processor need, hardware portability, ternary model quality or unicorn business.

## Artifacts / Logs

To be created when implementation begins, under a focused location such as:
- \`analysis/edge_inference/\` — streaming and measurement harness
- \`results/edge_inference/\` — manifests, raw predictions/traces, analysis
- \`tests/\` — protocol invariants and output-parity tests

Do not commit licensed videos or pretrained binaries. Preserve exact download instructions, hashes, and license notices.

## Follow-up Questions

1. Does the chosen data subset provide the per-frame 2D ground truth actually required for person-detection/event scoring?
2. Which hardware, OS, and external meter are available, and which device power boundary can be measured fairly?
3. What real customer duty cycle, model class, resolution, deadline and economic bottleneck determine a commercially relevant workload?
4. How much of steady-state device power is fixed decode/idle overhead, and therefore impossible to remove via inference scheduling?
5. Which observed bottleneck, if any, warrants event-driven reuse or ternary compute?

## Primary References

- AI City 2026 Track 1: https://www.aicitychallenge.org/2026-track1/
- AI City 2021 evaluation (different counting task): https://www.aicitychallenge.org/2021-data-and-evaluation/
- Torchvision SSDLite: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- ONNX Runtime EPs: https://onnxruntime.ai/docs/execution-providers/
- MLCommons inference power: https://docs.mlcommons.org/inference/power/
- MLPerf Tiny streaming benchmark motivation: https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/
- Prior FlexTern evidence: EXP-0005, EXP-0007, EXP-0008, EXP-0009/0009A; docs/THESIS.md.
