# Review — FT-0010 Control Adequacy Gate (EXP-0010B): plan and completion

Reviewer: advisor agent (Opus 5.5), consulted twice by the implementing Claude agent: once after orientation and a draft plan (before any code), once after the first commit eafa7e7 (results, records and handoff on disk). The briefs asked for independent challenge of the metric, the holdout discipline, overclaiming and provenance.
Date: 2026-11-04
Scope: `analysis/control_gate/`, `tests/test_control_gate.py`, `results/control_gate/`, `docs/experiments/EXP-0010B-control-adequacy-gate.md` and the canonical documents.

A review is advice, not an accepted change. Dispositions are at the end.

## Plan review (before implementation)

1. **Check the 3D boxes and calibration** for a whole-person label and an occlusion measure, validating on unoccluded persons. 2. **A foot point from a visible-extent box is not robust** (occlusion is at the bottom); use projected 3D foot points; consider ground-plane zones. 3. **Presence is degenerate;** the bar must beat trivial predictors (and fresh layouts need non-trivial base rates and vehicles). 4. **576x320 mixes aspect and resolution;** add an equal-pixel aspect-correct variant, recompute anchors, re-verify parity. 5. **Selection must require real-time operation.** 6. **Check YOLOX preprocessing** before scoring. 7. **Look at forklift false positives** before calling them confusion. 8. **Treat the fresh holdout as one shot** with a new locked protocol file and a declared failure outcome. 9. Guard optimized postprocessing with parity checks; defer INT8. 10. WSL2 cannot see real clock speed. 11. Record the metric as a Proposed ADR; add tests; persist the review.

## Completion review (after the first commit)

1. **Blocking for interpretation: the zone metric has one-sided jitter tolerance.** A detection occupies a zone if its foot is within the margin of it, but a zone is NEG only if no GT foot is within the same margin, so a detection a few pixels across a boundary counts as a false alarm; ignored persons (small, foot outside the frame) also leave their zones NEG. This matches the holdout pattern (TPR >= 0.98, FPR 0.57-0.76). Requested a labelled post-hoc rescoring with ignored objects as DONTCARE and a larger NEG margin, a direct check of the "18% match a labelled Person" false alarms, and a unit test pinning the asymmetry.
2. **The "about 80% humanoids" headline is a detection-level share, not the operating-point effect** (cell-level FPR fell 0.59 -> 0.34 and 0.61 -> 0.39, about 40%); lead with operating-point numbers; replace the untested "L10-style foot noise" attribution.
3. **Manifest provenance is wrong:** `git_info` checks dirtiness only under `analysis/edge_inference`, so every control-gate manifest says commit b056ace, clean, while the code was untracked and edited between runs. Add a code-state section; state that the faulty YOLOX development artifacts were deleted.
4. Wording: the faster decode/postprocess path was planned unconditionally (omission, not deferral); YOLOX tau = 0.05 is at the grid edge; the YOLOX "reference" preprocessing was written by the implementer, so the real evidence is the detector working.
5. Closeout: write this file, record the commit hash in both the experiment record and the handoff, do not cite a regression count from a guessed module list, push, list out-of-repo side effects.
6. **Correction of the reviewer's own earlier advice:** the plan review recommended choosing holdout layouts with vehicles present and never mentioned humanoids, which contributed to the class-mix gap.
7. Recommendation for the founder report: the pipeline is trustworthy for timing, the quality control is not; the evidence does not yet justify Stage B; run one corrected-gate objective (humanoid- and ignore-aware metric, tolerant NEG band, new fresh holdout whose layout rule screens class mix).

## Disposition (implementer)

- **Plan review:** all items adopted (3D projection validated per scene with a preregistered rule; per-scene failure of L09 handled; S2 equal-pixel variant added; anchors recomputed and parity-checked; feasibility rule in the selection; YOLOX preprocessing checked, which found a real bug (amendment 1); vehicle crops audited; protocol locked before holdout video download; holdout one-shot; INT8 not run; host clock sampled and thermal state recorded as unobserved; ADR-0007 Proposed; 18 tests). Not adopted as advised: ground-plane zones (image grid kept, disclosed) and an optimized decode/postprocess path (omitted; recorded as an omission).
- **Completion review 1:** confirmed. Rescoring run (`gt_states_tolerant`, `posthoc_tolerant_metric.json`): YOLOX-Tiny at the frozen tau balanced accuracy 0.844 / 0.863, FPR 0.306 / 0.253 (from 0.592 / 0.614); person-matched false alarms are almost entirely people with their foot outside the frame (2,114 of 2,128 and 1,501 of 1,502); FPR stays 0.25-0.31 at TPR >= 0.97 over all taus. The preregistered verdict is unchanged. Unit tests added (20 total).
- **2:** rewritten in the experiment record, CURRENT_WORK, PROJECT_STATE, the handoff and KI-015 with operating-point numbers first; the untested attribution was replaced by the R6 result.
- **3:** added "Code state of the recorded runs" to EXP-0010B; manifests were not rewritten (no rerun). The `git_info` limitation is documented, not fixed.
- **4:** corrected as listed.
- **5:** done except as follows: the broad regression job was stopped and no count is cited; the commit hashes are recorded in a follow-up docs commit.

## What the reviewer did not do

The reviewer did not re-execute any experiment and relied on the implementer's transcript and artifacts. Its first-round claim about 3D-box usability was disproved by the implementer's own validation (projected cuboids are loose), and the implementer then used a calibration-derived standard-height person instead.
