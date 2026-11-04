# EXP-0010B — Control Adequacy Gate (FT-0010)

Status: Complete (2026-11-04). **Result: NO ADEQUATE CONTROL AMONG THE TESTED CONFIGURATIONS** under the preregistered zone-occupancy bar on two fresh layouts. Stage B (temporal execution) is NOT started.
Date: 2026-11-04
Commit: eafa7e7 (results and records), fcdfbb2 (review fixes, post-hoc tolerant-metric rescoring, code-state section, review file)
Parent: `EXP-0010-edge-inference-baseline.md` (Stage A, preserved unchanged).

## Research Question

Is there a detector/input configuration that is an adequate quality control for fixed-camera industrial person monitoring, under an application-level metric, on layouts never used for any choice, while sustaining 30 fps on the CPU baseline? The answer is yes/no for the **tested configurations only**.

## Protocol (frozen before any new-configuration inference)

`analysis/control_gate/protocol_v2.json` (Stage A's `protocol.json` is untouched). One amendment, `protocol_v2_amendment1.json` (YOLOX input convention, see Findings 3), was made after development detections existed and before any holdout detection; `protocol_v2.json` itself is byte-identical to the frozen file.

**Tasks distinguished.** Person *detection* (boxes + IoU; needs box-extent agreement), *presence* (degenerate here: only 278/9000 L09 and 45/9270 L11 frames have no person), *count* (needs separation of crowded/occluded people), *zone occupancy* (is someone standing in region Z). **Primary = zone occupancy**: 3x3 image grid (stand-in for operator ROIs), position = ground-contact point (GT: projected 3D box bottom, validated per scene; detection: box bottom-centre), POS/NEG/DONTCARE with a 32 px boundary margin, balanced accuracy = (TPR + 1 − FPR)/2 so any constant predictor scores exactly 0.5.

**Quality bar (TARGET; no customer basis, not a safety standard)**, on every fresh layout: TPR ≥ 0.90, FPR ≤ 0.10, lower 95% block-bootstrap bound of balanced accuracy ≥ 0.85; plus ≤ 1% drops and p99 ≤ 100 ms at 30 fps. Declared failure outcome: "no adequate control among the tested configurations"; no third candidate or re-tuning on the holdout.

**Data roles.** L09, L10, L11 are all *development* (L11's Stage A "holdout" status is withdrawn; recorded as spent). Fresh holdout = layouts L08 (Warehouse_016 Camera_0002) and L02 (Warehouse_010 Camera_0002), chosen from ground-truth statistics alone by a written rule (geometry-consistent, median person ≥ 60 px, vehicles in ≥ 20% of frames, ≥ 3 non-trivial zones), 9000 frames each, never viewed before the one scoring.

**Candidates** (all ONNX Runtime CPU): S0 = Stage A SSDLite320 (320x320 squash); S1 = same weights at 576x320 aspect-preserving (1.8x pixels); S2 = 416x240 aspect-preserving (about the same pixels as S0; separates aspect from resolution); YT = YOLOX-Tiny 416 letterbox; YS = YOLOX-S 640 (not run, see Limitations). **License review:** YOLOX code and its release weights are Apache-2.0 (KNOWN, LICENSE of the Megvii-BaseDetection repository; trained on COCO 2017). YOLOv8/YOLO11/YOLOv10 are AGPL-3.0 and were not evaluated.

## Method

Code: `analysis/control_gate/` (`gt3d.py` 3D→2D projection; `consistency.py` geometry validation; `zones.py` metric; `cag_eval.py` scoring; `detectors.py` candidates; `parity.py`; `select_holdout.py`; `run_cag.py` CLI; `fp_crops.py`; `timing_summary.py`). Tests: `tests/test_control_gate.py` (18). Raw artifacts: `results/control_gate/` (manifests, per-frame replay traces, dev/holdout JSON, parity, geometry, FP audit images). Per-detection files live in the git-ignored cache and are referenced by hash in each run manifest.

Development screening used every 3rd frame of each clip (stride 3). Operating threshold per candidate = best pooled development balanced accuracy on the geometry-passing clips (L10, L11). Selection rule: among real-time-eligible candidates, highest pooled development balanced accuracy, ties within 0.01 to the cheapest.

## Code state of the recorded runs

Every manifest records `git.commit = b056ace` and `dirty = false`, which is **wrong for this objective**: `common.git_info` checks cleanliness only under `analysis/edge_inference`, and all of `analysis/control_gate/` was untracked and was edited between runs. What changed during the work: `detectors.yolox_pre` (raw BGR 0-255 after the first YOLOX run; regenerated), `parity.py` (reference input, a guard for the incomplete YOLOX-S file), `run_cag.py` (replay/feasibility/select/holdout commands were added after the first detect runs), `zones.py` (the post-hoc `gt_states_tolerant` was appended after all holdout scoring). The preregistered scoring functions (`zones.gt_states`, `det_states`, `cag_eval.evaluate`, `run_cag.cmd_holdout`) were unchanged from the first development scoring through the holdout scoring. The code state is commit fcdfbb2 (eafa7e7 plus the post-hoc tolerant metric and its test). Faulty-YT development artifacts were deleted (see Findings 3).

## Results

All MEASURED unless marked. No energy was measured.

### R1. Ground-truth geometry (MEASURED; `consistency.py`)
The 3D boxes + calibration reproduce the visible boxes for persons only in some scenes. For persons wholly inside the frame, a 1.75 m person at the projected foot has visible-height / expected-height median **1.00 (L11)**, 0.99 (L10) but **0.69 (L09)**, with foot-point errors under 15 px on L10/L11. Rule (preregistered): median ratio in [0.9, 1.1] and median foot offsets ≤ 15 px. **L09 fails** and contributes no zone metric. The 3D cuboids themselves are loose (projected whole-box width ≈ 1.5x the visible width on L09), so a projected 3D box is **not** a usable whole-person 2D label; a calibration-derived whole-person box (standard height + median unoccluded aspect) is used instead and labelled INFERRED. On L10 that INFERRED box does not agree with detector heights (YOLOX heights 1.28x expected), so whole-box numbers there are unreliable.

### R2. Parity (MEASURED; `results/control_gate/parity.json`)
S0/S1/S2: torch-vs-ORT heads ≤ 4.4e-5; detections at score ≥ 0.3 equal to the official torchvision model in 40/40 frames each; max box deviation 0.6 / 1.3 / 0.9 px (resize implementation). YT: preprocess identical to an independently written reference; detections at score ≥ 0.3 equal to the release's own `demo_postprocess`/`multiclass_nms` in 40/40 frames (15/40 non-empty), box deviation 0.0.

### R3. Development screening (stride 3; pooled over L10+L11 for the zone metric)

| config | tau | pooled balanced acc. | TPR | FPR | AP50 L10 / L11 (visible boxes) | AP at IoU 0.1 L10 / L11 | mean pipeline ms excl. decode (L11, 4 thr) |
|---|---|---|---|---|---|---|---|
| S0 | 0.10 | 0.691 | 0.519 | 0.136 | 0.204 / 0.129 | 0.246 / 0.341 | 16.1 |
| S1 | 0.10 | 0.763 | 0.702 | 0.177 | 0.369 / 0.037 | 0.562 / 0.582 | 25.2 |
| S2 | 0.10 | 0.740 | 0.580 | 0.100 | 0.224 / 0.042 | 0.287 / 0.461 | 16.7 |
| **YT** | 0.05 | **0.800** | 0.699 | 0.099 | **0.732 / 0.715** | 0.760 / 0.801 | 40.1 |

(Stage A S0 numbers differ slightly from the row above for L11 because the score floor here is 0.01, not 0.001, and only every third frame is scored.) Per clip, YT zone balanced accuracy is 0.926 on L11 (FPR 0.011) but only 0.646 on L10 (TPR 0.48, FPR 0.18); no candidate reaches the 0.90/0.10 target pooled. Count error is far above a constant-median predictor for S0/S1/S2 (e.g. L11 MAE 16–30 vs 1.6); YT is near it (1.31 vs 1.61 on L11, 0.87 vs 0.54 on L10).
Visibility analysis (L11, YT, detection hit at IoU ≥ 0.5 vs visible fraction): 0.16 (< 0.5) / 0.33 (0.5–0.8) / 0.57 (0.8–0.95) / 0.80 (≥ 0.95); for S0 0.10 / 0.05 / 0.21 / 0.42. Even essentially unoccluded people are matched only 42% of the time by S0 at IoU 0.5 but 69% at IoU 0.1 and S0 boxes are 0.4–0.8x the expected whole height on occluded people — **localization/extent disagreement (including the 320x320 squash, since S2/S1 improve IoU-0.1 AP most) is a large part of S0's loss, and detector capacity is the rest**: YT's AP50 is 3.5–5.5x S0's on the same frames.

### R4. Real-time feasibility (preregistered rule, 60 s paced replay of L11 frames 0–1799, queue 2; `feasibility.json`)
S0, S1, S2 pass at 4 threads (0 drops; p99 43 / 49 / 40 ms). YT fails at 4 threads (10.5% dropped, p99 112 ms) and passes at 8 (0.2%, p99 54 ms). **Selected by the rule: YT, tau 0.05, 8 threads** (0.800 vs next best 0.763).

### R5. Fresh holdout (ONE scoring each at the frozen tau; `holdout_hold_L08.json`, `holdout_hold_L02.json`)

| layout | config (tau) | balanced acc. [95% CI] | TPR | FPR | AP50 (visible) | bar |
|---|---|---|---|---|---|---|
| L08 | S0 (0.10) | 0.705 [0.679, 0.732] | 0.980 | 0.571 | 0.305 | **not met** |
| L08 | YT (0.05) | 0.701 [0.672, 0.732] | 0.993 | 0.592 | 0.790 | **not met** |
| L02 | S0 (0.10) | 0.616 [0.586, 0.645] | 0.996 | 0.763 | 0.107 | **not met** |
| L02 | YT (0.05) | 0.682 [0.650, 0.714] | 0.979 | 0.614 | 0.621 | **not met** |

**Verdict under the declared rule: no adequate control among the tested configurations.** TPR is high; the failure is false alarms.

### R6. Post-hoc diagnostics (labelled POST-HOC: run after seeing the holdout results; they do not change the verdict)
- **Tau does not rescue the preregistered metric.** Best balanced accuracy over every tau on the holdout is only 0.724 (L08, YT) and 0.684 (L02, YT) (`posthoc_tau_sweep.json`).
- **Humanoid robots (operating-point effect).** Setting zones that contain a humanoid foot to DONTCARE moves YT cell-level FPR at the frozen tau from 0.592 to 0.338 (L08) and from 0.614 to 0.387 (L02): roughly 40% of the false-alarm rate, balanced accuracy 0.828 / 0.794 (`posthoc_humanoid_dontcare.json`). (At the detection level the share is larger: of YT confident, score >= 0.5, false-alarm detections, 82% (L08) and 81% (L02) overlap FourierGR1T2/AgilityDigit labels at IoU >= 0.3, `posthoc_fa_by_class.json`; that is a count of detections, not of the cell-level rate.) The development layouts contain no humanoids, so development could not reveal this. Stage A treated humanoids as ignore regions; **I did not carry that policy into the zone metric and my layout rule did not screen humanoids.** An independent advisor review had also advised choosing holdouts "where forklifts are present" without mentioning class mix, which contributed.
- **Metric construction (found in review).** The preregistered metric gives foot jitter tolerance on the miss side only: a detection counts as occupying a zone if its foot is within 32 px of it, but a zone is NEG only if no GT foot is within the *same* 32 px, so a detection 10 px across a boundary from a nearby person is a false alarm. Persons that are ignored (under 32 px, foot outside the frame, bad depth) also leave their zones NEG. Pinned by `tests.test_control_gate.TestMetricToleranceAsymmetry`. Of the confident false alarms that match a labelled Person, 2,114 of 2,128 (L08) and 1,501 of 1,502 (L02) are persons whose projected foot is **outside the frame** (people cut off at the bottom edge; the detector clips their box to the frame, the metric has no GT foot there).
- **Ignore-aware, tolerant rescoring (`gt_states_tolerant`, `posthoc_tolerant_metric.json`, `posthoc_tolerant_tau_sweep.json`)**: humanoid feet and all ignored persons make their zones DONTCARE and NEG requires no GT foot within 2x the margin. At the frozen tau, YT: balanced accuracy 0.844 (L08, TPR 0.993, FPR 0.306) and 0.863 (L02, TPR 0.979, FPR 0.253); S0: 0.834 (FPR 0.313) and 0.741 (FPR 0.514). Over all taus YT's best is 0.852 (L08) and 0.863 (L02), with FPR about 0.25-0.31 at TPR >= 0.97 and never below 0.15 at TPR >= 0.67. **Still below the bar (FPR <= 0.10 at TPR >= 0.90).** The source of the remaining FPR at high recall was not isolated (candidates: low-score detections on clutter, residual ground-truth foot noise, unlabelled or ambiguous persons).
- Development with the tolerant variant: L11 is unchanged (YT 0.926 -> 0.930), L10 improves slightly (0.646 -> 0.686, TPR still 0.48).
- **Forklift/pallet-truck audit** (30 crops, `results/control_gate/fp_audit/dev_L10_S0_tau0.1_vehicle_fp.jpg`): the "forklift" false positives of S0 on L10 are a static pallet truck at the dock and a blue autonomous mobile robot; no person is on or hidden by them, so they are **detector confusion, not label semantics**. YT produces 22 such detections vs S0's 1,528 on the same clip.

### R7. Controlled timing (hold_L08 frames 0–2699, AC power, Balanced plan; interleaved S0_t4, YT_t8, S0_t8, YT_t4 x3; `timing_summary.json`)

| config | p50 ms (3 reps) | p99 ms | max ms | dropped | mean stage ms decode / pre / network / post | busy cores |
|---|---|---|---|---|---|---|
| S0, 4 thr | 18.9, 19.1, 25.3 | 29.4, 29.4, 37.6 | 42–69 | 0 | 5.2 / 0.7 / 9.0 / 4.5 | 3.5–4.0 |
| S0, 8 thr | 17.5, 17.5, 17.6 | 26.9, 26.9, 27.3 | 42–85 | 0 | 5.6 / 0.7 / 6.1 / 5.3 | 7.6 |
| YT, 4 thr | 90.9, 98.3, 92.9 | 115.9–126.7 | 141–162 | **12.2%, 19.9%, 13.1%** | 6.6 / 1.0 / 35.8 / 1.0 | 4.3–4.4 |
| YT, 8 thr | 28.7, 28.9, 37.8 | 39.1, 42.5, **99.4** | 63–197 | 0, 0, 0.78% | 5.3 / 0.9 / 21.8 / 1.1 | 7.9–8.3 |

YT at 8 threads meets the real-time bar only marginally (p99 99.4 ms and 0.78% drops in the third repeat). The third repeat of S0-4, YT-8 and the others is slower while the host reported 3488 MHz at start of some of them: the Windows-reported clock (2499/3488 MHz, `CurrentClockSpeed`) changed between runs, WSL2 cannot observe frequency, and thermal state is unobserved, so run-to-run differences (up to 6 ms in p50, one 197 ms max) are **confounded by boost/thermal/host state and not attributable to the configuration**. No trace violations in any run. CPU busy cores are diagnostic only (ORT threads spin) and are **not** an energy estimate.

### R8. Optimized conventional comparators: what was and was not done
Measured: ORT FP32 with a 4-vs-8 intra-op thread comparison on two detectors (R7). **Not done, as deviations from what protocol_v2 planned:** a faster decode/resize path, fused in-graph postprocessing, OpenVINO/hardware decode (protocol_v2 planned the faster decode/postprocess path unconditionally; I did not get to it, so this is an omission, not a rule-based deferral). INT8/FP16 quantization was conditional on the FP32 control passing and was therefore not run. Decode plus postprocess are still 5-12 ms (more than the network for S0), so the baseline remains optimized for the network only. No FlexTern advantage is claimed or tested.

## Findings and negative results

1. **No adequate control was found** (R5). The preregistered answer is "no among {S0, YT (selected), also S1/S2/YS untested on the holdout}".
2. **A stronger detector helps a lot on boxes but does not close the zone gap**: YOLOX-Tiny AP50 0.62-0.79 on fresh layouts vs 0.11-0.31 for the Stage A detector, and TPR >= 0.98. The preregistered metric's FPR (0.59 / 0.61) is inflated by its own construction: humanoid robots (about 40% of the cell-level FPR), people cut off at the frame edge and the one-sided boundary tolerance (R6). With those removed (post-hoc) FPR is still 0.25-0.31 at the same recall; its source is unexplained.
3. **Preprocessing bug caught before the holdout (negative result):** the YOLOX-Tiny weights give person scores ~1e-3 under the release demo ImageNet mean/std normalization and 0.7-0.8 with raw 0-255 pixels; the first development run therefore scored exactly 0.500 (no detections). My original parity check validated against that same mismatched demo preprocessing and passed trivially (counts equal at zero). Fixed to raw BGR 0-255 (amendment 1) and regenerated. The reference input in the parity check is written by me, so it only confirms the code matches my intent; the real evidence that the convention is right is the detector working (AP50 about 0.73 on L10/L11 and 0.62-0.79 on fresh layouts). The faulty first-run development artifacts were deleted; only their summary numbers (4 detections >= 0.05 in 3090 frames, pooled balanced accuracy 0.500) survive in amendment 1.
4. **My torchvision `fixed_size` mistake:** it is (width, height); my first S1/S2 parity reference ran transposed and reported false disagreements. Pipeline was correct; the check was fixed.
5. **Methodology gaps found after holdout scoring (R6):** the zone metric ignored the humanoid ignore-region policy, ignored persons left NEG zones, boundary tolerance was one-sided, and the layout rule did not screen class mix. The one-shot holdout is therefore *spent for this protocol*; any future pass claim needs a **new fresh holdout** and a corrected protocol version.
6. **L09 fails the geometry check**, so zone metrics exist on two development clips only (L10, L11) and the pooled development score is dominated by whichever of those has more frames.
7. **Timing is not cleanly separable from host state** (R7).

## Interpretation

- **INFERRED:** with the metric as preregistered, neither SSDLite variant nor YOLOX-Tiny is an adequate control; with the metric corrected post-hoc (humanoids and ignored persons DONTCARE, symmetric tolerance), YOLOX-Tiny is closer (balanced accuracy 0.84-0.86, TPR >= 0.98) but its FPR (0.25-0.31) is still far above the 0.10 target, and what limits it was not isolated.
- **INFERRED:** squash vs aspect is a smaller effect than detector capacity; S2 (same pixels, correct aspect) improved IoU-0.1 AP over S0 (0.46 vs 0.34 on L11) but not IoU-0.5 AP.
- **INFERRED:** a real 30 fps single-stream CPU budget on this laptop is about 8 threads for a YOLOX-Tiny-class model; the SSD-class model leaves most of the frame period idle.
- **HYPOTHESIZED, untested:** the remaining false alarms are low-score detections on clutter plus residual label/foot-projection mismatch and unlabelled or ambiguous persons; a corrected protocol on a new holdout would show whether YOLOX-Tiny clears the bar. YT tau = 0.05 sits at the lower edge of the grid, so its optimum may lie below it.

## Surprises

1. About 40% of the holdout false-alarm rate (80% of confident false-alarm detections) came from a COCO "person" detector detecting humanoid robots that the labels deliberately exclude, and more came from the metric's own construction.
2. The same detector that scores 0.93 zone balanced accuracy on L11 scores 0.65 on L10 and ~0.70 on fresh layouts: layout variance dominates detector choice.
3. A trivially passing parity check hid a preprocessing bug.

## Limitations

- Synthetic data; four fresh/dev layouts with usable geometry; one camera per layout; ~5-minute clips; frames strongly correlated (block bootstrap).
- Zone metric: 3x3 image grid, 32 px margin, foot point from a 3D-box projection that is itself only accurate to ~15 px median and noisier in tails; the bar is a TARGET without a customer.
- Candidate set is small; YOLOX-S was not evaluated (its download stalled and was stopped by a memory reaper; it was designated optional), no fine-tuning, tiling or larger model; YOLOX-Tiny's pipeline excludes decode/postprocess optimization.
- Timing: one laptop CPU under WSL2, AC power, Balanced plan, frequency/thermal unobserved, 90 s runs x3 on one clip (the real-time feasibility rule used a development clip, L11).
- Two background jobs were killed by the host's memory protection (the development-detection loop, and the L02 holdout run) and restarted with the founder's authorization. Detections already completed before a kill were reused as written (S0 development detections; L08 S0/YT and L02 S0 holdout detections); nothing was partially reused.
- **No physical power or energy was measured.**

## What This Proves

- **MEASURED:** under the preregistered zone-occupancy bar, S0 and the selected YOLOX-Tiny both fail on two fresh synthetic layouts (balanced accuracy 0.62–0.71), mainly through false alarms.
- **MEASURED:** YOLOX-Tiny is a much better box detector than the Stage A model on this data (AP50 0.62–0.79 vs 0.11–0.31 on fresh layouts) but needs 8 CPU threads for 30 fps (about 2.3x S0's per-frame latency).
- **MEASURED (post-hoc):** about 40% of the holdout zone false-alarm rate is on humanoid robots excluded from the labels, and more of the metric's inflation is its one-sided boundary tolerance and people cut off at the frame edge; with all three removed YOLOX-Tiny still has FPR 0.25-0.31.

## What This Does NOT Prove

Nothing about real industrial footage, a customer's quality requirement, INT8/FP16 or ternary quality, power/energy, or any FlexTern advantage. It does not prove that no adequate control exists, only that the tested ones, under this preregistered metric, did not meet this bar.

## Follow-up Questions

1. Should the application metric treat humanoid robots as persons, ignore them, or report them separately (a customer-semantics question for the founder; ADR-0007 proposes ignoring)? How should people cut off by the frame edge be handled?
2. With a corrected, humanoid-aware protocol and a *new* fresh holdout, does YOLOX-Tiny (or YOLOX-S) clear the bar? Is the remaining FPR (0.25-0.31 post-hoc) label noise or detector error?
3. Are the 3x3 image-grid zones and the foot-projection ground truth adequate, or should zones be ground-plane regions?
4. Is the quality gap worth closing before temporal-execution work, or should temporal experiments proceed against YOLOX-Tiny with the metric caveats stated?
5. Real footage, a real deadline, and an energy meter (Q-026) remain unavailable.

## Artifacts / Logs

`analysis/control_gate/` (code, `protocol_v2.json`, `protocol_v2_amendment1.json`); `results/control_gate/`: `holdout_selection.json`, `parity.json`, `dev_scores.json`, `feasibility.json`, `dev_selection.json`, `holdout_hold_L08.json`, `holdout_hold_L02.json`, `timing_summary.json`, `posthoc_*.json`, `fp_audit/*.jpg`, `runs/<role>/<config>/.../{manifest,summary,stage}.json` and per-frame `trace.csv.gz` for replays. Review: `docs/reviews/REV-2026-11-04-ft-0010-control-adequacy-gate-review.md`. Dataset attribution for the committed crops: NVIDIA PhysicalAI-SmartSpaces (CC-BY-4.0; Wang et al., Physical AI Smart Spaces, per the dataset card).
