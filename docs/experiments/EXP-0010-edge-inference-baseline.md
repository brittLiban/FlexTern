# EXP-0010 — Continuous edge-inference baseline (FT-0010)

Status: Stage A implemented and measured (2026-11-04). The reference and deployment runtimes agree exactly and the harness is reproducible, but the pretrained detector scores poorly under the frozen IoU 0.5 metric and a post-hoc diagnostic shows the cause is mixed (box-extent mismatch and confusion with non-person objects), so it is not yet a validated control. Stage B is NOT started; awaiting founder + ChatGPT review.
Date: 2026-11-04
Commit: a3c9ce6

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

## Stage A — Preregistered Protocol (only)

Build a reproducible streaming-video *baseline* and validity audit before testing FlexTern enhancements. The refined, frozen protocol (splits, scoring policy, queue/drop policy, tolerances) is `analysis/edge_inference/protocol.json`, written before any model ran; its SHA-256 is in every run manifest.

1. **Dataset selection / rights audit.** Primary candidate: AI City Challenge 2026 Track 1 Physical AI Smart Spaces, single RGB fixed-camera views with per-frame 2D labels; use a selected subset of train/validation scenes and keep scenes disjoint across calibration/development/final holdout. Confirm actual files, class coverage, rights, annotation format, timestamps, and train/validation split before training or scoring. The corpus is synthetic: DO NOT call the result a real-factory validation.
2. **Task restriction.** Initial task is **person detection/occupancy on one camera** where a compatible pretrained detector can supply a legitimate baseline. Forklifts, pallet trucks and robot classes require separately demonstrated model coverage or fine-tuning. Multicamera 3D-HOTA is the challenge's original task, not our Stage A metric; report that benchmark mismatch explicitly.
3. **Reference model.** Torchvision SSDLite320-MobileNetV3-Large COCO weights. A second small detector (e.g. YOLOv8n) is a later cross-model test, with model license review.
4. **Execution.** PyTorch FP32 correctness reference (B0) and ONNX Runtime CPU (B1), streaming every frame at the *source* FPS with identical decode, resize, preprocessing, NMS, threshold and output schema. Fixed CPU threads, explicit warmup. No GPU required.
5. **Scoring.** Verified ground-truth 2D boxes and a fixed IoU/confidence policy for person AP50, F1 and recall; report unmatched/unknown frames instead of inventing labels. Event/episode recall only after ground-truth entry/exit episodes and acceptable detection deadline are explicitly defined and audited.
6. **Logging.** For each frame: source timestamp, arrival/processing/output timestamps, decode/preprocess/inference/postprocess time, predictions, dropped frames. For each run: dataset/model SHA-256, Git commit, environment, threads, power state, repeats, duration.
7. **Energy boundary.** Gross J/camera-hour is a later physical metric requiring external power measurement at a documented device boundary. Early telemetry (CPU utilization, latency) is *diagnostic only*; no estimated energy, PyTorch speed, FLOP reduction or Yosys cell count may be presented as measured device energy.
8. **Baseline validation.** At least two disjoint camera scenes and repeated runs; model parity checked before timing; report the magnitude of repeat-run variance. Once a final holdout is locked, do not tune on it.

**Stage A definition of done:** another engineer can retrieve the documented dataset subset, inspect the environment/manifest, run a full-every-frame reference and one optimized CPU inference implementation, reproduce per-frame outputs and timing, and see an explicit statement of all metrics **not yet measured**. No gain claim is necessary.

## Later comparisons (not current commitments)

After the founder reviews Stage A, possible next objectives are conventional MOG2 gating; detector every N=2/4/8 frames plus tracking; a validated event scheduler; oracle temporal bound; activation caching; hardware-supported FP16/INT8 with representative calibration; only then a properly trained/fine-tuned ternary control. One objective is selected at a time, consistent with AGENTS.md.

Important controlled comparisons:
- Temporal approach vs **best conventional gating/tracking** at equal downstream event semantics.
- Ternary + temporal vs **INT8 + identical temporal strategy**, not ternary vs naive FP32.
- Gross physical energy, not only inference latency or skipped MACs.
- Paired camera-level trials, not independently sampled frames from the same clip.

## Baselines

B0: full-frame PyTorch FP32 eager reference (quality/correctness only). B1: full-frame ONNX Runtime CPU, same model, same end-to-end pipeline. Later: B2 supported INT8 with calibration; B3 hardware-specific EP; B4/B5 motion gate and keyframe+tracker.

## Method (as executed)

**Code:** `analysis/edge_inference/` (README there lists every module and the reproduce commands); tests `tests/test_edge_inference.py`. **Raw artifacts:** `results/edge_inference/` (`summary.md` is the generated roll-up; per-run `manifest.json`, `summary.json`, per-frame `trace.csv.gz`). Per-detection files (score >= 0.001, 1.2–1.9 M rows per clip, tens of MB) are in the git-ignored cache and referenced by SHA-256 in each `summary.json`; they regenerate deterministically (see Determinism).

### Data (all MEASURED from the dataset files unless labeled)

- Source: Hugging Face `nvidia/PhysicalAI-SmartSpaces`, `MTMC_Tracking_2026`. License **CC-BY-4.0** (dataset card; KNOWN). Ungated, downloadable without login. Data are **synthetic** (Omniverse/IsaacSim, plus Cosmos-Transfer re-renderings); `Warehouse_026/027` are real-world test captures but ship **videos and calibration only, no ground truth** (60 s each) and are therefore unusable for quality scoring; they were not used.
- Per-scene `ground_truth.json` (15–280 MB, all cameras, 2D "visible" boxes + 3D boxes). **Finding:** the real key names contain spaces (`"object type"`, `"2d bounding box visible"`) while the dataset README documents underscores; the parser accepts both. Every frame index has an annotation record (no unlabeled frames); no per-box visibility/occlusion fraction is provided.
- Video: H.264 High, 1920x1080, 30 fps, **no B-frames**, constant PTS step 1/30 s, keyframe every 30 frames; decoded frame count equals the GT frame count on all three selected clips.
- **Layout audit (`results/edge_inference/dataset_audit.json`).** The 23 GT scenes (000–022) collapse to 12 *layout groups* by identical camera geometry or identical `map.png` bytes. The README's Cosmos-Transfer pairs (001/003/005/007/021/024 vs 000/002/004/006/020/023) are confirmed. **Finding:** scenes 000–007 and the nominal *validation* scenes 020/021 share one layout (L00), so the nominal train/val split is not layout-disjoint; Stage A splits are therefore assigned by layout group, not by scene name.
- Splits (frozen before any model ran; chosen from ground-truth statistics and GT-overlay inspection only):

| role | layout group | scene / camera | frames | GT person boxes (all / evaluated >=32 px) | GT person height px p5/25/50/75/95 |
|---|---|---|---|---|---|
| calibration (threshold only) | L10 | Warehouse_019 / Camera_0002 | 9000 | 18612 / 16093 | 7 / 121 / 214 / 348 / 406 |
| development | L09 | Warehouse_017 / Camera_0001 | 9000 | 10975 / 10892 | 107 / 148 / 200 / 259 / 350 |
| holdout (reported) | L11 | Warehouse_022 / Camera_0002 | 9270 | 50044 / 48670 | 61 / 96 / 123 / 154 / 195 |

  Each clip is 5.0–5.15 minutes, one camera. Class vocabulary in the dataset: Person, Forklift, PalletTruck, NovaCarter, Transporter, FourierGR1T2, AgilityDigit; only Person is scored (a COCO detector cannot be a valid control for the others). No humanoid classes occur in the selected clips; the code treats them as ignore regions (tested). The dataset's own task (multi-camera 3D HOTA) is **not** what is scored here.

### Scoring policy (frozen)
Person AP50 (101-point COCO AP, IoU 0.5, detections >= 0.001); GT person boxes under 32 px native height are ignore regions (a detection matched to one, or unmatched and under 32 px, is neither TP nor FP); the all-size AP is also reported. F1/precision/recall at one operating threshold chosen **only on the calibration clip** (grid 0.05–0.95, max F1): **tau = 0.15** (calibration F1 0.270). Dropped frames output nothing and count as misses. CIs: 95% percentile block bootstrap over contiguous 10 s (300-frame) blocks, 1000 resamples, seed 20261104 (frames are not treated as independent).

### Model and pipeline
`ssdlite320_mobilenet_v3_large`, COCO_V1 weights, checkpoint SHA-256 `a79551df90c7...5d65d2` (matches torchvision's filename hash); ONNX (opset 17, fixed 1x3x320x320, network only) SHA-256 `e668b54dc2f3...a29dc`; anchors SHA-256 `bfea495927de...55c45`. Shared numpy/OpenCV preprocessing (bilinear 1920x1080 -> 320x320, non-aspect-preserving, mean/std 0.5) and person-only postprocessing (softmax, score > 0.001, top-300, torchvision BoxCoder, NMS 0.55); only the network differs between B0 and B1. B1: ORT 1.31.1 CPUExecutionProvider, ORT_ENABLE_ALL, 4 intra-op threads. **Correction (found in review):** the B1 processes of all runs in R3 (offline and replay reps 0-2 and 10-12) *did import torch* before timing started, because the manifest code located the checkpoint through `torch.hub`; this was fixed afterwards (`models.checkpoint_path` no longer imports torch; test `test_manifest_and_identity_do_not_import_torch`). It inflated B1's recorded memory (see R3) and should not affect timing (an imported-but-unused torch starts no work; one torch-free 60 s replay, `replay_r20`, gave p50 23.2 ms / p99 50.0 ms on battery, inside the earlier ranges, but that is one short run). B0: torch 2.11.0+cu128 eager, 4 threads, `inference_mode`.

### Streaming harness
Frame i is released at t0 + pts_i (perf_counter); a feeder thread decodes (PyAV, 2 slice threads) at release; decoded frames enter a FIFO of capacity 2 (oldest dropped when full); a worker runs pre -> network -> post. Latency = output_done − *scheduled* release (so decode, queueing and feeder slip all count). The encoded MP4 is loaded in memory first. 50 untimed warmup inferences. Every frame has exactly one trace row (including dropped ones); trace invariants are validated after every run (no violations in any run). The 100 ms deadline is **illustrative** (≈3 frame periods); no customer deadline is known.

### Code state of the recorded runs
Every manifest records git commit `bb30579` with `dirty = true`: the runs executed from an uncommitted working tree (the Stage A code was written during the session and committed afterwards). From the calibration run onward, `pipeline.py`, `evaluate.py`, `replay.py` and `models.py` (other than the torch-free `checkpoint_path` fix above, which changes no computed value) and the frozen `protocol.json` were not modified. After the calibration run these changed: `run_stage_a.py` (`host_state` parsing, a holdout guard) and `report.py`, which affect metadata and reporting only; `diagnose_fp.py` was added. The final Stage A commit contains exactly this code; its hash is a3c9ce6.

### Host (recorded in every manifest)
AMD Ryzen 7 260 (8C/16T), 15.7 GB RAM (WSL2 capped to 6 GB), WSL2 Ubuntu 26.04 guest (kernel 6.18.33.2-microsoft-standard-WSL2) under Hyper-V on a laptop, Windows "Balanced" plan. Governor, frequency, thermal state and process priority are **not controlled**. AC/battery: see Findings (power-state confound).

## Results

All numbers below are **MEASURED** unless labeled; none is an energy measurement.

### R1. Parity and determinism (before any timing claim)
- Raw head tensors, torch vs ORT, 500 development frames: max |Δcls_logits| 3.7e-5, max |Δbbox| 4.3e-5 (tolerance 1e-3). AP50 on those frames: official `model(img)` 0.2142, shared-pipeline B0 0.2140, B1 0.2140. Same detection count (score >= 0.3) in 500/500 frames, official vs shared and B0 vs B1; B0-vs-B1 max box deviation 1.5e-4 px; official-vs-shared max deviation 8.4 px at 1080p (resize implementation: OpenCV uint8 vs torch float).
- Documented deviation quantified: the official path caps 300 detections across all classes, so it returns 41,931 person boxes where the person-only path returns 80,310 on the same 500 frames; AP differs by 0.0002.
- Determinism: two offline B1 runs on each of development and holdout produced bit-identical detections (equal SHA-256).
- Alignment: scoring the B0 development detections against GT shifted by −3..+3 frames peaks at 0 (AP50 0.2046 at 0; 0.2036 at −1; 0.2023 at +1). The peak is shallow (tolerance of IoU 0.5 for slowly moving people), so this is a consistency check, not a sub-frame proof; GT overlays on decoded frames were also inspected visually (frames 100/4000/9000 of the holdout, 500/5000 of development).

### R2. Detection quality (offline, every frame, tau = 0.15)

| clip | AP50 (95% CI) | AP50 no size floor | precision | recall | F1 (95% CI) | recall small / medium / large |
|---|---|---|---|---|---|---|
| calibration L10 (threshold chosen here) | 0.202 [0.153, 0.256] | 0.175 | 0.476 | 0.188 | 0.270 [0.213, 0.328] | 0.00 / 0.00 / 0.21 |
| development L09 | 0.204 [0.172, 0.240] | 0.203 | 0.591 | 0.159 | 0.251 [0.211, 0.289] | 0.00 / 0.00 / 0.16 |
| **holdout L11** | **0.130 [0.101, 0.163]** | 0.126 | 0.230 | 0.207 | 0.218 [0.191, 0.243] | 0.00 / 0.03 / 0.26 |

B0 and B1 are identical to all printed digits on development and holdout (the calibration clip was run with B0 only). Frame-level: a person-containing frame has >=1 true positive in 20% (development) / 60% (holdout) of frames. Frame-level false alarms (>=1 FP at tau on a frame with no scored person): 0 of 278 person-free frames (development), 0 of 45 (holdout), undefined for calibration (all 9000 frames contain a scored person); those denominators are small. Small/medium/large = native height < 32 / 32–96 / >= 96 px; "small" boxes are ignore regions in the primary metric.

### R2b. Post-hoc false-positive / box-semantics diagnostic (cached detections, no new inference; `analysis/edge_inference/diagnose_fp.py`, `results/edge_inference/fp_diagnostic.json`)
The dataset's boxes are the **visible extent** of each object; a COCO detector predicts the whole person. To separate that from a genuinely failing detector, the same B0 detections were re-scored at looser IoU and the false positives at tau = 0.15 were classified by their best overlap with any ground-truth box. The frozen primary metric (IoU 0.5) is unchanged; the holdout row is explicitly **post-hoc** and nothing was tuned on it.

| clip | AP at IoU 0.1 / 0.3 / **0.5** | F1 at IoU 0.1 / 0.3 / 0.5 | FP: near-miss person (IoU 0.1-0.5) | FP on non-person GT (forklift, pallet truck, ...) | duplicate of a matched person | background (IoU < 0.1 with everything) |
|---|---|---|---|---|---|---|
| calibration | 0.244 / 0.222 / **0.202** | 0.302 / 0.280 / 0.270 | 11% | **88%** | 1% | 0.2% |
| development | 0.324 / 0.281 / **0.204** | 0.312 / 0.289 / 0.251 | **50%** | 25% | 11% | 15% |
| holdout (post-hoc) | 0.342 / 0.260 / **0.130** | 0.359 / 0.310 / 0.218 | **68%** | 0.1% | 8% | 23% |

(FP counts: 3,326 / 1,198 / 33,868.) Reading: **(MEASURED)** at IoU 0.5 the holdout score is dominated by localization/extent disagreement: loosening IoU to 0.1 raises AP 2.6x and F1 from 0.22 to 0.36, and two thirds of its false positives overlap a person box by 0.1-0.5. **(MEASURED)** the calibration clip is different: 88% of its false positives sit on forklifts/pallet trucks, i.e. genuine class confusion. Even at IoU 0.1 the holdout reaches only recall 0.34 / precision 0.38, so the detector is still weak; whether the near-misses reflect tight visible-extent labels on occluded/crowded people, the 16:9 -> 320x320 squash, or detector localization quality was **not separated**.

### R3. Real-time replay at source frame rate (B1, ORT CPU, 4 threads; 9000/9270 frames per run)

| clip | power at start | repeats | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | dropped / frames | frames over 100 ms |
|---|---|---|---|---|---|---|---|---|
| development | battery* | 3 | 22.4–24.3 | 30.8–34.4 | 41.6–47.3 | 55.9–134.6 | 0–3 / 9000 | 0–6 |
| development | AC | 3 | 20.1–21.9 | 28.8–31.3 | 37.4–**61.3** | 79.9–**838.6** | 0–**28** / 9000 | 0–27 |
| holdout | battery* | 3 | 22.2–23.8 | 33.7–36.3 | 48.8–50.7 | 59.7–120.2 | 0–1 / 9270 | 0–4 |
| holdout | AC | 3 | 19.7–21.4 | 29.5–34.9 | 41.3–50.7 | 113.8–124.3 | 0–1 / 9270 | 2–4 |

(Ranges are min–max across the repeats; each repeat is a fresh process. `*` = state recovered from the raw string of the first-version host-state parser; see Findings.) Mean stage times, B1, per frame: decode 4.7–7.0 ms, preprocess 0.8–1.0 ms, **network 9.3–11.1 ms**, postprocess 4.6–6.8 ms. Peak RSS as recorded 818–865 MB, **but those B1 processes had torch loaded (see Method); a torch-free B1 process (60 s holdout replay, `replay_r20`, battery) started at 246 MB and peaked at 315 MB** — use the latter as the ORT pipeline footprint (it still includes the in-memory encoded clip, ~100 MB, plus Python/OpenCV/PyAV; B0 peaks at 1.03–1.04 GB); average 3.75–3.95 busy cores of 16 (diagnostic only; ORT intra-op threads spin-wait, so this overstates work). Feeder slip p99 0.26–2.4 ms (the harness pacing is accurate). Quality including drops equals offline quality to 3 digits on every B1 run (AP50 0.2045 development, 0.1296 holdout).

B0 (PyTorch eager, 1 repeat per clip, battery*): network 46–47 ms per frame; it cannot keep up with 30 fps on 4 threads: **38% (development) and 39% (holdout) of frames dropped**, p50 latency 115 ms, p99 149–152 ms, 4762–4794 processed frames over the 100 ms deadline; quality including drops AP50 0.131 (development) / 0.084 (holdout) against 0.204 / 0.130 offline. It is a correctness reference, not a competitor, and this is not an architecture comparison.

Offline (no pacing, includes decode): B1 41–43 fps, B0 15–17 fps (INFERRED from stage times: the B1/B0 network time ratio is about 4.4x, which is a statement about PyTorch eager vs ORT, not about any FlexTern technique).

## Findings and negative results

1. **The baseline scores poorly under the frozen metric, for mixed reasons (MEASURED; cause partly isolated).** Holdout AP50 0.13, F1 0.22; medium-height persons (32-96 px) are almost never matched at IoU 0.5 (3% recall) and large ones 16-26% at the operating point. The post-hoc diagnostic (R2b) shows two different failure modes: on the calibration clip 88% of false positives are fires on forklifts/pallet trucks (genuine confusion; also seen in two overlay frames); on the holdout two thirds of false positives are near-misses on real people (IoU 0.1-0.5), so a large part of its low score is box-extent/localization disagreement with visible-extent labels, and AP rises 2.6x at IoU 0.1 (still only 0.34). Squashing 16:9 to 320x320 and a COCO-to-synthetic domain gap are HYPOTHESIZED contributors; neither was tested. No tiling, higher resolution, other detector or fine-tuning was tried (out of Stage A scope; the preregistration forbids silent method changes). An earlier draft of this record called the detector simply "weak"; that is not supported as a single-cause statement.
2. **The operating point does not transfer across scenes (MEASURED).** tau = 0.15 gives precision 0.48 / 0.59 / 0.23 on calibration / development / holdout.
3. **Power-state confound found during the work (MEASURED, my error).** The first-version host-state parser stored the raw battery code glued to the power-plan string and labeled it "unknown". Recovering it showed that only the calibration run was on AC and **every other run (including all first-set replays) started on battery** (code 1, "discharging"), Balanced plan. I repeated the B1 replays on AC (reps 10–12) rather than overwrite anything. AC medians are ~2.6 ms lower on average (holdout p50 19.7–21.4 vs 22.2–23.8; development 20.1–21.9 vs 22.4–24.3), but the AC set contains the worst tails (839 ms stall and 28 drops in one run), so AC vs battery is **not cleanly separable** from thermal state and host scheduling noise. The parser is fixed (`battery_status_code`, charge %, plan name are separate fields in later manifests). Latency numbers must be quoted as ranges across both sets.
4. **Unexplained stalls (MEASURED, cause UNKNOWN).** 7 of 12 B1 replays contain at least one frame >100 ms (max 839 ms) although the stage costs are ~20 ms; feeder slip stayed small in all runs, so the harness clock was not the cause. Candidates (HYPOTHESIZED, untested): WSL2/Hyper-V scheduling, Windows background activity, thermal/boost behavior, Python GC. The 839 ms run overlapped with light interactive host activity and a tool timeout, so it is retained but flagged.
5. **"Optimized ONNX Runtime" applies to the network only.** Preprocessing, softmax/box decoding and NMS are shared unoptimized numpy; postprocess (4.6–6.8 ms) costs about half the network (9.3–11.1 ms) and decode another 4.7–7.0 ms, so the network is only ~45% of per-frame compute. B1 is therefore *not* the strongest possible deployment baseline (e.g. softmax/decode/NMS could be fused into the graph or run on faster kernels; hardware video decode and lower-resolution decoding were not tried). Any later comparison against it must say so.
6. **Environment defect.** The PyPI `torchvision 0.26.0` wheel does not match `torch 2.11.0+cu128` in the WSL venv (`operator torchvision::nms does not exist` at import); the matching `+cu128` wheel from the PyTorch index fixes it. Recorded in KNOWN_ISSUES.
7. **README/dataset mismatches (MEASURED):** ground-truth key names; nominal val scenes 020/021 share a layout with train scenes 000–007.
9. **My claim "torch never imported" was false for B1 (found by the independent review).** The manifest code imported torch to locate the checkpoint, so every recorded B1 run carried a ~500 MB torch footprint; corrected above and in R3, with a regression test.
8. **Harness bugs found and fixed during development** (not results): container closed before reading stream attributes; test fixtures with an impossible "empty" softmax; Windows console encoding in the report writer.

## Interpretation

**INFERRED, with the limits below:**
- The measurement chain (data alignment, shared pre/post, torch-vs-ORT parity, determinism, trace validity, drop accounting) is sound enough to use for **CPU latency/drop baselines** on this clip family.
- On this 8C/16T laptop CPU with 4 threads, an ORT SSDLite320 pipeline sustains 30 fps with median end-to-end latency ~20–24 ms and p99 ~37–61 ms, using about two thirds of one 33.3 ms frame period per frame (mean stage sum ~21–24 ms), of which the network is under half. Whether skipping inference would save useful energy is **not established**: it is a statement about stage shares of latency, not about power, and ORT's spinning threads make CPU utilization unusable as a proxy.
- The *quality* baseline is not yet established as a control for temporal-execution experiments: under the frozen metric the detector matches only ~16-26% of large persons, and R2b shows the score is entangled with box semantics (visible-extent labels, IoU criterion) and, on some scenes, class confusion. Retention claims such as "event recall relative to ground truth" would inherit both. The immediate question is therefore partly *what the application metric is* (localization-tight boxes vs presence/count/zone occupancy), not only which detector to use; that question is open (Q-025).

## Surprises

1. Pretrained COCO SSDLite320 reaches only AP50 0.13-0.20 at IoU 0.5 on mostly large synthetic workers, and loosening IoU to 0.1 lifts the holdout to 0.34 (box semantics are a large part of the story).
2. The nominal validation scenes of the dataset share a layout with training scenes.
3. A silent host-power-state confound existed until the AC repeat set; AC did not give uniformly better tails.
4. Non-network stages are about as expensive as the network.

## Limitations

- Synthetic footage; two evaluated layouts (plus one calibration layout), **one camera per layout, ~5 minutes each**; frames are strongly correlated (blocks, not frames, are the unit of uncertainty; block CIs from 30–31 blocks are wide); holdout results come from one scene.
- Person class only, 2D boxes, no event/episode metric (entry/exit episodes and deadlines were not defined or audited), no identity/tracking.
- The primary metric is IoU 0.5 against visible-extent boxes; R2b is post-hoc and uses cached detections only; its holdout row is labeled post-hoc.
- Size-stratified scoring depends on the frozen 32 px floor; GT "visible" boxes carry no occlusion fraction, so some scored persons are heavily occluded.
- One CPU, one OS configuration (WSL2 on a laptop), 4 threads, no thread sweep, no governor/thermal control, power state confounded (Finding 3). No GPU, OpenVINO, INT8, FP16 or hardware-decoder baseline (B2/B3 not run).
- B0 replay has one repeat per clip; calibration had no replay or B1 offline run; development B0 offline was run once.
- The 100 ms deadline and queue depth 2 are illustrative choices, not customer requirements.
- **No physical power or energy was measured.** CPU utilization is diagnostic only.
- Detections are stored for score >= 0.001 and AP uses them; absolute AP is sensitive to that floor (the official default).

## What This Proves

- **MEASURED:** a reproducible, tested pipeline exists that replays encoded video at source rate with full per-frame logging, produces bit-identical outputs across two runtimes' parity checks and across repeated offline runs, and reports quality with block-bootstrap uncertainty.
- **MEASURED:** pretrained COCO SSDLite320 FP32 scores holdout AP50 0.13 (IoU 0.5; 0.34 at IoU 0.1) on the selected synthetic warehouse clips, whatever the runtime; R2b shows the causes are mixed and not fully separated.
- **MEASURED:** ORT CPU sustains 30 fps on this machine at p50 20–24 ms with rare drops; PyTorch eager does not (38–39% drops).

## What This Does NOT Prove

No energy or power result; no FlexTern speedup or temporal-execution gain; nothing about real industrial footage, other cameras, other classes, other CPUs or accelerators; no statement about INT8/FP16/ternary quality; no evidence that a better-tuned detector will behave like this one; no business or product-market claim.

## Artifacts / Logs

- `analysis/edge_inference/` — harness; `analysis/edge_inference/protocol.json` — frozen protocol; `analysis/edge_inference/README.md` — reproduce commands.
- `results/edge_inference/summary.md|json` — generated roll-up; `dataset_audit.json`; `identity.json` (model/clip hashes); `parity.json`; `alignment_sweep.json`; `calibration.json` (tau grid); `runs/<clip>/<runtime>/<mode>_r<k>/{manifest,summary}.json` + `trace.csv.gz`; `alignment/*.jpg` GT overlays.
- `results/edge_inference/alignment/*.jpg` are annotated, downscaled frames from NVIDIA's PhysicalAI-SmartSpaces dataset (CC-BY-4.0; Wang et al., Physical AI Smart Spaces, NeurIPS 2026, arXiv:2610.02580, per the dataset card), kept only as alignment evidence.
- Not committed (licensed/large): videos, ground-truth JSON, per-detection npz, checkpoint, ONNX file. Re-fetch from the Hugging Face URLs; hashes are in `identity.json` and the manifests.
- `tests/test_edge_inference.py` — 42 tests (31 run on host Python, all 42 in the WSL venv); with the 182-test non-RTL baseline in one host process: 224 OK, 11 skipped.
- Diagnostic: `analysis/edge_inference/diagnose_fp.py`. Review: `docs/reviews/REV-2026-11-04-exp-0010-stage-a-review.md`; handoff: `docs/handoffs/2026-11-04-exp-0010-stage-a.md`.

## Follow-up Questions

1. Which detector/input configuration is an *adequate* control on this data (a preregistered AP50/recall bar on the same holdout), and is it available under acceptable licenses?
2. Is the optimized baseline's non-network 60% (decode/preprocess/postprocess) reducible, and what would the strongest practical conventional pipeline cost at the same quality?
3. Which hardware, OS and external meter will measure gross device power, and which boundary is fair?
4. What real customer duty cycle, model class, resolution and deadline determine a commercially relevant workload? Does the synthetic corpus resemble it at all?
5. What causes the occasional 100–840 ms stalls, and is it removable on a dedicated Linux host?
6. How much of the near-miss mass at IoU 0.1-0.5 is label semantics (visible extent) vs the squash vs detector localization, and what application-level metric (presence, count, zone occupancy, tight boxes) does a customer need? (R2b separates class confusion from near-misses but not these three.)

## Primary References

- AI City 2026 Track 1: https://www.aicitychallenge.org/2026-track1/
- Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces (CC-BY-4.0)
- AI City 2021 evaluation (different counting task): https://www.aicitychallenge.org/2021-data-and-evaluation/
- Torchvision SSDLite: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- ONNX Runtime EPs: https://onnxruntime.ai/docs/execution-providers/
- MLCommons inference power: https://docs.mlcommons.org/inference/power/
- MLPerf Tiny streaming benchmark motivation: https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/
- Prior FlexTern evidence: EXP-0005, EXP-0007, EXP-0008, EXP-0009/0009A; docs/THESIS.md.
