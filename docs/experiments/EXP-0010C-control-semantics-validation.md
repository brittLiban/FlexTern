# EXP-0010C — Control Semantics and Residual Error Validation (FT-0010)

Status: Complete (2026-11-04). **Result under the preregistered decision rule: CHANGE_WORKLOAD.** Policy B (person-shaped-agent occupancy) passes on one fresh layout and narrowly fails on the other; policy A (human-only) fails on both; no single score threshold meets the gating pair on all six clips (post-hoc). Stage B (temporal execution) is NOT started and is not recommended on this evidence.
Date: 2026-11-04
Commit: 97cbd9b (protocol lock), e9772df (results and records), ce2803d (completion-review fixes); fresh runs executed from 97cbd9b with analysis/ and tests/ clean
Parents: `EXP-0010B-control-adequacy-gate.md` and `EXP-0010-edge-inference-baseline.md` (both preserved unchanged).

## Research Question

Can a trustworthy industrial-vision quality control be established for YOLOX-Tiny without repeatedly changing models and evaluation criteria? The answer is for the tested detector, metric and layouts only.

## Protocol

`analysis/control_semantics/protocol_v3.json`, locked by `results/control_semantics/lock.json` (committed in 97cbd9b before any fresh video was downloaded or any detector ran on a fresh clip; run_cs refuses fresh-holdout fetch/detect/score unless the lock is committed, matches the protocol hash and `analysis/`/`tests/` are clean). The lock fixes the policies, the ground-plane metric and its parameters, the per-policy operating thresholds taken from development data, the gating bar (a TARGET without a customer basis), the **decision rule mapping outcomes to a recommendation**, the fresh-holdout layout rule and the no-fine-tuning constraint.

### Two applications, not one
- **A — human-only monitoring** (safety zones, lone-worker analytics in a site that also contains humanoid and other robots): humanoid robots are **negatives**; a detection on one is a false alarm. A COCO person detector has no robot class, so it is expected to fail A unless a model can separate humans from humanoid robots.
- **B — person-shaped-agent occupancy** (generic presence analytics, or a conservative policy that treats every person-shaped agent as a human): Person, FourierGR1T2 and AgilityDigit are positive. Forklifts, pallet trucks, AMRs and transporters are negatives under both.
Neither policy ignores humanoids to raise a score; both are computed on identical detections and each belongs to its own commercial application. General machine/vehicle occupancy needs a different class set and is out of scope.

### Metric (ground-plane, derived from measurement)
Cells of 3 m on the floor; positions from the 3D box centre (GT) and the back-projected box bottom-centre (detections); tolerance 0.75 m = rounded maximum 95th-percentile localization error (R2). POS / NEG / DONTCARE as in `gp_metric.py`: NEG requires no positive-class agent (scored or ignored) within the cell grown by 2 x margin (the geometric symmetric counterpart of the prediction tolerance); truncated or small agents are ignored and truncated or small detections are excluded and counted (symmetric rule). Gating pair: **TPR >= 0.90 and false alarms per predicted occupied cell <= 0.10**, plus bootstrap bounds (TPR lower >= 0.85; precision lower >= 0.85) on every fresh layout, plus the real-time bar. Cell-level FPR and balanced accuracy are reported but not gating because FPR is diluted by empty cells (29 to 671 cells across development clips).

## Method and data roles
Development (diagnosis, tau, parameters): L10, L11, and L08, L02 (scored once and inspected in EXP-0010B). L09 excluded from zone metrics (geometry rule). **Fresh holdout, chosen from ground truth alone by the locked rule:** L05 (Warehouse_013 Camera_0004) and L03 (Warehouse_011 Camera_0000); L06 and L07 remain unused in reserve. Detector: YOLOX-Tiny 416 letterbox, raw BGR 0-255 (EXP-0010B amendment 1), ORT CPU, 8 threads, no change. Code: `analysis/control_semantics/`; tests `tests/test_control_semantics.py` (21). Raw artifacts: `results/control_semantics/`.

## Results

All MEASURED unless marked; nothing here is an energy measurement.

### R1. Residual false alarms of EXP-0010B's metric, attributed by the triggering detection (POST-HOC, development data; `fa_attribution.json`)
EXP-0010B's image-grid false alarms (reproduced: FPR 0.184 / 0.011 / 0.592 / 0.614 on L10 / L11 / L08 / L02 at tau 0.05), each assigned to the highest-scoring detection that triggered the cell and classified by that detection's own box against all ground-truth classes with one fixed priority order (humanoid, truncated detection, ignored person, person at a zone boundary, person foot mismatch, partial person overlap, other object, low-score clutter, unexplained):

| clip | humanoid | frame-edge truncated detection | person at zone boundary | vehicle/AMR (IoU >= 0.3) | low-score clutter | unexplained (score >= 0.3, no overlap) |
|---|---|---|---|---|---|---|
| L08 | 61.8% | 24.6% | 5.3% | 3.9% | 0.3% | **0** |
| L02 | 70.0% | 10.6% | 10.9% | 3.6% | 0.3% | **0** |
| L10 | 0 | 15.9% | 28.4% | 0.3% | 46.1% | 0 |
| L11 | 0 | 49.5% | 46.3% | 0 | 0.5% | 0 |

(Corrected classifier `fa_attribution_v2.json`; the first version, kept in `fa_attribution.json`, lumped partial humanoid/vehicle overlaps into an `other object` bucket of 6.2% / 6.6% / 1.0% / 0 for L08 / L02 / L10 / L11. The remaining shares are person-at-boundary/foot-mismatch/partial, ignored persons, humanoid-partial 2-3% and vehicle-partial under 1% on L08/L02.)

**Reading:** the EXP-0010B "residual FPR 0.25-0.31" was mostly humanoid and frame-edge-truncated detections that its projected-foot rescoring failed to neutralize (the advisor's suspicion, confirmed). No confident detection was left unexplained, so there is no evidence of unlabelled people among the false alarms. Genuine detector errors in this metric are the small vehicle/AMR share on L08/L02 and the low-score clutter on L10 (see R6-R8).

### R2. Foot localization in metres (POST-HOC; `localization.json`)
Matched, unoccluded persons (IoU >= 0.5): ground error median / 90th / 95th percentile = 0.37 / 0.65 / 0.70 m (L10), 0.20 / 0.44 / 0.56 m (L11), 0.26 / 0.51 / 0.76 m (L08), 0.25 / 0.44 / 0.52 m (L02); the same foot error in pixels has a median of 8.5-32.8 px across those layouts. The tolerance is therefore physical (0.75 m), not pixel.

**R2b (added after the completion review; POST-HOC, `localization_humanoid.json`).** The 0.75 m margin was derived from Person rows only, but policy B scores humanoids with it. Humanoid ground error (matched, IoU >= 0.5): FourierGR1T2 median 0.17-0.19 m, 95th percentile 0.48 / 0.68 m (L08 / L02), 2.5-4.2% beyond 0.75 m (like persons); **AgilityDigit median 0.33 m, 90th percentile 1.1-1.3 m, 95th percentile 1.3-1.6 m, 24-26% beyond 0.75 m**. The locked margin is unchanged, but it does not hold for AgilityDigit: part of policy B's humanoid position-mismatch false alarms is humanoid geometry/localization noise, not detector error.

### R3. Is the 3x3 image grid representative? No (MEASURED, `localization.json`)
The nine image-grid cells cover floor areas from 1.6 x 2.8 m (bottom row) to 12.6 m x 161 m (top row, L10) within one frame, and the same cell differs by up to about 7x between layouts in one dimension (top-row depth 161 m on L10 vs 24 m on L11; bottom-row cell 2.8 x 2.0 m on L10, 4.3 x 4.0 m on L11); one image row is 0.004-0.048 m of floor depending on row and layout, so a fixed 32 px margin is about 0.1-1.5 m vertically. Ground-plane regions are necessary for a tolerance with one physical meaning. The v2 image-grid result is still reported as written below, for comparison only.

### R4. Class mix and humanoid geometry (GT only, before the policies; `survey.json`)
Humanoid robots are present in essentially every frame of every unused layout (L01, L03-L07: humanoid frame fraction >= 0.97 in 33 of 36 cameras; the exceptions are L05 cameras 3 and 5 (0.76, 0.50) and L03 camera 5 (0.25)); only the development layouts L09-L11 have none. Forklifts/pallet trucks are present in L03 only among the unused layouts (L01, L04-L07: 0%). Humanoid 3D geometry passes the person-style rule in 18 of the 36 unused-layout cameras (person geometry passes in 15; both in 9). The policy question therefore decides the outcome on almost all remaining data.

### R5. Miss attribution (POST-HOC; `miss_attribution.json`, tau 0.02, ground tolerance 0.75 m)
Agent-level recall of scored persons: **unoccluded (visible fraction >= 0.95): 0.81-0.94; heavily occluded (< 0.8): 0.01-0.39 (L10 0.11-0.14, L11 0.01, L02 0.00-0.04)**; by visible height: 32-64 px: 0.00-0.01; 64-128 px: 0.16-0.67; >= 128 px: 0.75-0.94. The recall limit is occlusion and small apparent size at 416 px input (a 64 px person is about 14 px after the letterbox resize), i.e. genuine detector limits, not labelling.

### R6. Development scores and thresholds (`dev_policy_scores.json`)
At the locked tau = 0.02 (macro balanced accuracy; both policies chose 0.02) policy B on development clips: TPR 0.78 / 0.83 / 0.99 / 0.87 and precision 0.21 / 0.93 / 0.92 / 0.47 (L10 / L11 / L08 / L02). Development already fails the gating pair on three of four clips. **Flaw in what I locked:** macro balanced accuracy is diluted by empty cells, so it chose the lowest threshold, where low-score clutter dominates false alarms on some clips (R7). The lock was not changed after seeing this; the fresh result stands as locked.

### R7. Fresh holdout, ONE scoring each at the locked tau 0.02 (`fresh_hold_L05_YT.json`, `fresh_hold_L03_YT.json`)

| layout | policy | TPR [95% CI] | precision (false alarms per predicted cell) [95% CI of precision] | FPR | balanced acc. | bar |
|---|---|---|---|---|---|---|
| L05 (17 cells) | **B** | 0.992 [0.989, 0.995] | 0.958 (**0.042**) [0.934, 0.976] | 0.168 | 0.912 | **met** |
| L05 | A | 0.984 [0.978, 0.989] | 0.212 (0.788) [0.165, 0.261] | 0.620 | 0.682 | not met |
| L03 (185 cells) | **B** | 0.919 [0.905, 0.931] | 0.828 (**0.172**) [0.811, 0.846] | 0.019 | 0.950 | **not met** (precision) |
| L03 | A | 0.950 [0.936, 0.962] | 0.431 (0.569) [0.410, 0.450] | 0.068 | 0.941 | not met |

Detections used / excluded as truncated or short: L05 146,082 / 83,349 (36% of retained-size detections), L03 193,617 / 14,129. Ground-truth agents scored / ignored (B): L05 41,798 / 96,835 (L05 camera 4 is heavily truncated, so its pass rests on a minority of agents in 17 cells); L03 150,885 / 16,936.
**Coverage (what the metric scores).** Ground-truth agents scored / ignored under B (ignored = truncated at the frame edge, under 32 px, or projected outside the frame): L05 41,798 / 96,835 (**70% ignored**, 17 cells), L03 150,885 / 16,936 (10%); development: L10 10,245 / 8,367 (45%), L11 39,894 / 10,150 (20%), L08 60,648 / 81,601 (57%), L02 113,117 / 60,824 (35%). On L05, 36% of tall detections were excluded as touching the frame edge (L03: 6.8%; the reported count `excluded_truncated_or_short` is tall edge-touching detections only, shorter detections are dropped uncounted). **The metric validates only the image interior; people cut off at the frame edge, often the nearest and most safety-relevant, are DONTCARE. "B passes L05" therefore means: on 17 interior cells and the 30% of agents that are not truncated.**
**EXP-0010B's v2 image-grid metric as written (tau 0.05):** L05 balanced accuracy 0.715 (TPR 0.998, FPR 0.568), L03 0.883 (TPR 0.971, FPR 0.206). The metric change **flips L05 from fail to pass for B**; it is a metric effect, disclosed, not a detector improvement. Person AP50 (visible boxes): 0.779 (L05), 0.887 (L03).

**Preregistered decision rule applied:** B fails on L03 (precision), so not PROCEED_TEMPORAL; B has TPR >= 0.70 and false alarms <= 0.30 on both fresh layouts (majority), so **CHANGE_WORKLOAD**. Policy A fails on both layouts: the human-only application needs a model objective (human/humanoid discrimination), not a different evaluation.

### R8. Post-hoc checks after the fresh scoring (they do not change the verdict)
- **No threshold meets the pair everywhere** (`tau_sweep_posthoc.json`, policy B, tau 0.02-0.8): per clip, taus meeting TPR >= 0.90 and false alarms <= 0.10 exist only for L08 (0.02-0.2) and L05 (0.02-0.3); none for L10, L11, L02, L03; none for all six at once. (L03 reaches false alarms 0.092 at tau 0.05 but then TPR is 0.867.)
- **What remains of B's false alarms, corrected classifier (`fa_attribution_gp_v2.json`, `fa_attribution_v2.json`; supersedes the first version in `fa_attribution_gp.json`, which is kept).** The first classifier lumped partial overlaps with humanoids and vehicles into one `other object` bucket (62% on L05, 61% on L08); the corrected split at the locked tau (share of false-alarm cells):

| clip | vehicle/AMR/transporter (IoU >= 0.3) | vehicle/AMR partial | humanoid partial | humanoid position mismatch | person position mismatch | person partial | low-score clutter | unexplained | FA/pred if both mismatch buckets were label noise |
|---|---|---|---|---|---|---|---|---|---|
| L10 | 2% | 4% | 0 | 0 | 3% | 4% | **88%** | 0 | 0.765 |
| L11 | 0 | 0 | 0 | 0 | 19% | 18% | **63%** | 0 | 0.055 |
| L08 | 8% | 25% | 28% | 24% | 3% | 3% | 9% | 0 | 0.057 |
| L02 | 4% | 2% | 6% | 3% | 3% | 8% | **75%** | 0.01% | 0.501 |
| L05 | **53%** | 5% | 5% | 2% | 5% | 2% | 29% | 0 | 0.039 |
| L03 | 20% | 1% | 4% | **27%** | 7% | 4% | **37%** | 0 | **0.112** |

  L05 has NovaCarter/Transporter AMRs in every frame (survey), so its 53% is genuine vehicle/AMR confusion. On L03, even if both position-mismatch buckets were label noise, B would still have 0.112 false alarms per predicted cell, above the 0.10 limit, so the verdict does not depend on that noise.
- **Crop audit on L03 (the clip that decides the verdict; `fp_audit/hold_L03_B_*.jpg`, 12 crops per bucket).** Low-score clutter (scores 0.02-0.15): cardboard boxes on a conveyor belt, genuine detector false positives. Vehicle/AMR bucket: detections on white AMRs, yellow transporters and pallet loads, genuine confusion with non-person objects. Humanoid bucket: detections on the robots themselves whose box bottom does not land at the robot's labelled floor position (torso-only or partly occluded robots); detector category correct, localization or label geometry off (R2b). No crop showed an unlabelled person, so annotation problems are not a major cause of the remaining false alarms on this clip.
- **Genuine detector errors after the corrections:** low-score clutter on non-person objects (boxes, machinery) at the locked tau 0.02, confusion with AMRs/transporters/pallet loads, and occlusion/small-person misses (R5). **Not detector errors, or only partly:** detections on humanoids (class ambiguity, a policy matter) and the AgilityDigit/humanoid position mismatch (geometry noise).

### R9. Timing (short interleaved comparison, hold_L05 frames 0-2699; `timing_summary.json`) — ON BATTERY
The laptop was discharging (77-80%, Balanced plan) during these runs. **This is a deviation from protocol_v3**, which specified AC power for timing; it is also unlike the AC runs of EXP-0010B, so the numbers are **not comparable** with that timing and are confounded by power state; frequency and thermal state are unobserved under WSL2. The four 90 s replays can be rerun on AC if the founder plugs in; the quality verdict does not depend on them.

| config | p50 ms | p99 ms | dropped | network ms (mean) | busy cores |
|---|---|---|---|---|---|
| S0 SSDLite320, 4 thr | 23.9, 24.4 | 37.3, 36.7 | 0, 0 | 10.2 | 3.9-4.0 |
| YOLOX-Tiny, 8 thr | 48.9, 73.2 | 88.1, **108.9** | 0.04%, **0.85%** | 30.3-30.9 | 8.3 |

On battery YOLOX-Tiny at 8 threads misses the p99 <= 100 ms bar in one repeat and its network stage is about 8 ms (about 1.4x) slower than in the EXP-0010B AC runs. The real-time bar is therefore **not demonstrated** for YOLOX-Tiny in this session; it does not affect the quality verdict.

## Findings and negative results
1. **No trustworthy control was established** for the tested detector: the preregistered rule gives CHANGE_WORKLOAD (R7), and post-hoc no threshold serves four of six clips (R8). The two layouts where B meets the pair are large-person, humanoid-rich scenes.
2. **EXP-0010B's residual error was mostly metric, not detector** (R1): humanoids and frame-edge truncation. After the corrections the remaining errors are genuine detector limits: occlusion/small-person misses (R5) and, at the locked low tau, confusion with AMRs/transporters/pallet loads and clutter such as boxes on a conveyor (R8, crop audit), plus an AgilityDigit localization tail (R2b) that is partly label geometry.
3. **The human/robot question is the commercial crux.** Under A (humanoids are negatives) a COCO person detector fails everywhere (false alarms 0.57-0.79 on fresh layouts); under B it can pass on easy layouts. Which application is real is a founder decision; the evaluation does not choose.
4. **Metric changes move verdicts** (R7: v2 vs v3 on L05). The v3 metric is better justified (R2, R3) but the comparison shows how much of EXP-0010B's failure was metric construction.
5. **My locked threshold rule was flawed** (R6): balanced accuracy chose the lowest tau. Disclosed; not changed after the fact.
6. **WSL git vs the CRLF working tree:** git run from WSL without `core.autocrlf=true` reports every file as modified, which would have made a clean-repo guard unusable; provenance now pins that setting.
7. Hardware timing is power-state-sensitive (R9): the 8-thread detector's network stage was about 1.4x slower on battery in this session than in the AC runs of EXP-0010B (the two sessions also differ in time and thermal history, so the cause is not isolated).

## Interpretation
- **INFERRED:** the failure is layout-dependent recall (occlusion, small apparent size) and precision (vehicles, clutter, humanoids), and cannot be fixed by thresholding; closing it needs a model/input change (resolution, capacity, training with a robot class and the site's occluders), which is a separate objective with its own development/holdout split, not a further tuning of this one.
- **INFERRED:** a different workload definition (what occupancy requirement the detector actually serves) is a prerequisite for any temporal-execution experiment, because retention would otherwise be measured against a control that is wrong on most layouts in different ways.
- **HYPOTHESIZED, untested:** higher input resolution or a larger detector would raise recall on small/occluded people; a robot class would make A tractable.

## Limitations
- Synthetic data; two fresh layouts (one with 17 cells and mostly truncated agents); one camera per layout; ~5-minute clips; no real footage.
- The bar is a TARGET without a customer; margin, cell size and policies are my proposal (ADR-0008, Proposed).
- The ground-plane GT uses the 3D box centre, validated per scene (person rule, and a humanoid rule for the fresh clips) but not independently; L09 fails the person rule and is unused for zones. The 0.75 m margin was measured on persons only and does not hold for AgilityDigit (R2b).
- The v3 metric scores the image interior only: 70% of B agents on L05 and 35-57% on L08/L02 are DONTCARE (R7 coverage).
- Every EXP-0010C run manifest produced before the completion review records `command` as `python -m analysis.control_gate.run_cag ...` (the wrapped EXP-0010B builder hard-codes it); the real entry point is `python -m analysis.control_semantics.run_cs` with the same arguments. Those manifests are not rewritten; the wrapper now records the right command for future runs.
- tau was selected by a rule with a known flaw (R6). Post-hoc R7/R8 analyses used the fresh detections after the one scoring.
- Timing was measured on battery; energy and power are NOT measured; thermal state and true clock frequency are unobserved; no INT8/FP16, no fused or faster decode/postprocess (production-quality optimization is missing); CPU busy cores are diagnostic only.
- The EXP-0010C development diagnostics (`fa_attribution`, `localization`, `miss_attribution`, `dev_eval`, `survey`, `select_fresh`) are post-hoc JSON outputs without run manifests; they were produced by the code that is committed in 595ff77 (a small patch to `gp_metric.metrics_for` and `dev_eval` preceded the final `dev_eval` run). Fresh-holdout runs carry the whole-repository state (commit 97cbd9b, `dirty_code_paths` empty).
- EXP-0010B's manifests still record commit b056ace as clean; they are preserved unchanged and their code state is described in EXP-0010B.

## What This Proves
- **MEASURED:** under the locked ground-plane protocol YOLOX-Tiny meets the preregistered gating pair for policy B on one of two fresh layouts and fails on the other; for policy A it fails on both.
- **MEASURED (post-hoc):** no single threshold meets the pair on all six clips; the genuine residual errors are occlusion/small-person recall and confusion with vehicles/clutter, and no confident false alarm is unexplained.
- **MEASURED:** EXP-0010B's residual false alarms were mostly humanoid and frame-edge-truncated detections.

## What This Does NOT Prove
Nothing about real industrial footage, a customer requirement, other detectors or training, energy, thermal behavior, INT8/FP16, or any FlexTern advantage; nothing about whether a different detector or a trained model would pass.

## Recommendation (answering the founder's question)
**Change the application/workload; do not proceed to temporal execution; do not keep tuning this detector or this metric.** This is the plain reading of the preregistered decision rule (CHANGE_WORKLOAD), not an escalation to stop and not a softening. Rationale: (1) the locked rule yields CHANGE_WORKLOAD; (2) the failures are real detector limits that vary by layout (recall under occlusion and for small people; precision against AMRs, pallet loads and clutter) plus a human-versus-humanoid ambiguity that a person detector cannot resolve, and no threshold serves four of six clips; (3) retention experiments against this control would not be interpretable; (4) further tuning of thresholds, margins, cells or the person size/visibility range that YOLOX-Tiny happens to handle would be the repeated criterion-changing the founder ruled out, and "near-field cameras" is not a way out because the near-field people are the truncated ones that the metric ignores.
**What the workload change must be:** it has to come from a product or customer requirement that the founder sets (which application, A or B or another; which person sizes, visibilities and camera ranges matter), not from what this detector passes. Only then can it be (a) validated on a new fresh holdout, or (b) turned into a separate model objective (input resolution or capacity, a humanoid-robot class for application A) with its own development/holdout split. Stop the branch if that objective also fails a fresh holdout. This is a recommendation for the founder and ChatGPT, not a decision.

## Follow-up Questions
1. Which application is the product (A human-only, B person-shaped agents, or something else) and what minimum person size/visibility must be monitored?
2. Is a model-side objective (higher input resolution or a larger detector, a humanoid-robot class) justified, and under what budget and holdout?
3. Is real footage obtainable so that synthetic-layout variance is not the limiting evidence?
4. Energy: still no device and meter boundary (Q-026).

## Artifacts / Logs
`analysis/control_semantics/` (code, `protocol_v3.json`, README); `results/control_semantics/`: `lock.json`, `survey.json`, `localization.json`, `fa_attribution.json`, `miss_attribution.json`, `dev_policy_scores.json`, `fresh_selection.json`, `fresh_hold_L05_YT.json`, `fresh_hold_L03_YT.json`, `tau_sweep_posthoc.json`, `fa_attribution_gp.json` (first classifier, kept) and `fa_attribution_gp_v2.json` / `fa_attribution_v2.json` (corrected classifier), `localization_humanoid.json`, `fp_audit/hold_L03_B_*.jpg`, `timing_summary.json`, `runs/<role>/<config>/.../{manifest,summary,stage}.json` and replay traces. Review: `docs/reviews/REV-2026-11-04-ft-0010c-control-semantics-review.md`. ADR: `docs/decisions/ADR-0008-edge-control-policies-and-ground-plane-metric.md` (Proposed).
