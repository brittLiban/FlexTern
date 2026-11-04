# Review — EXP-0010 Stage A (continuous edge-inference baseline) completion

Reviewer: advisor agent (Opus 5.5, consulted by the implementing Claude agent after the implementation and a first draft of the record were on disk; the brief asked for an independent look at overclaiming, hidden confounds and whether the recommendation follows from the evidence — different from the implementation prompt).
Date: 2026-11-04
Scope: `analysis/edge_inference/`, `tests/test_edge_inference.py`, `results/edge_inference/`, `docs/experiments/EXP-0010-edge-inference-baseline.md` as drafted before this review.

A review is advice, not an accepted change. What the implementer did with each point is stated at the end.

## Overall Assessment

The harness and the record were judged strong: scene splits by layout group rather than by name, a frozen protocol written before any model ran, parity and determinism established before any timing claim, block-bootstrap uncertainty, a repeated run set and a candid limitations section. Four items needed correction before committing and one diagnostic needed to run before the final recommendation was written.

## Findings

1. **Blocking — B1 runs loaded torch.** The manifest builder located the checkpoint via `torch.hub`, so every B1 process imported torch before timing started. The record's "torch never imported" was false; B1 peak memory (818–865 MB against about 1.0 GB for B0) was inflated by roughly the size of torch; the existing no-torch test covered `environment()` but not `manifest()`. Requested: confirm, remove the import, re-measure memory in a torch-free process, correct the record, add a regression test, and say timing is probably unaffected rather than assume it.
2. **Run before finalizing the main finding.** The record left the 33,868 holdout false positives unexplained on a clip without forklifts. The ground truth is the *visible extent* of each object while a COCO detector predicts the whole person, so occlusion/crowding produces paired false-positive/miss events at IoU below 0.5 that look like low precision and low recall at once. Requested: a no-inference diagnostic on cached detections — AP and operating point at IoU 0.1/0.3/0.5, and the false positives at the operating threshold split into near-misses (best person IoU 0.1–0.5) versus fires on nothing (< 0.1) — on development and calibration, with any holdout use labeled post-hoc and the frozen metric untouched. If near-misses dominate, the next objective is closer to "fix the box semantics or the metric" than "swap the detector".
3. **Wrong statement.** The record said the frame-level false-alarm rate was "undefined" because every frame contains a person. Development has 278 person-free frames and the holdout 45; the summaries report 0.0 with real (small) denominators, and only calibration is undefined. "B0 and B1 identical on every clip" should have said development and holdout because calibration was run with B0 only.
4. **Unproven claim.** "Existing suites unchanged" had not been verified by running them. `analysis/model_precision` has top-level modules named `common` and `evaluate`; relative imports should avoid a clash, but it needed proof in one process.
5. **Hygiene before commit.** Manifests all record commit `bb30579` with a dirty tree — state that the runs came from an uncommitted tree and which result-bearing modules were unchanged from the calibration run onward; add the dataset's CC-BY attribution for the committed frames; remove a stray scratch directory; check Windows `git status`; report out-of-repo side effects (package installs, caches, the deleted Windows cache, the battery state, the killed watcher).

## Disposition (implementer)

- 1: confirmed (`'torch' in sys.modules` was True after `manifest()`); fixed in `models.checkpoint_path`; test added; a torch-free 60 s B1 replay measured 246 MB start / 315 MB peak; record corrected; timing treated as probably but not proven unaffected (one torch-free run inside the earlier range).
- 2: run. Result (EXP-0010 R2b): on the holdout (post-hoc) AP rises 0.13 -> 0.34 at IoU 0.1 and 68% of false positives are person near-misses; on the calibration clip 88% of false positives are on forklifts/pallet trucks; development is mixed (50% near-misses, 25% non-person). The record's single-cause "weak detector" statement was rewritten as a mixed-cause statement and the recommendation changed from a detector-adequacy gate to a control-adequacy gate that fixes the application metric first.
- 3: corrected in the record.
- 4: full non-RTL suite run in one host process (see the handoff for the count).
- 5: done as listed in the handoff, except that nothing in this review changed the experimental design.

## What the reviewer did not do

The reviewer did not re-execute any experiment and relied on the implementer's transcript and artifacts; its claims about code behavior (item 1) were verified by the implementer afterwards rather than taken on trust.
