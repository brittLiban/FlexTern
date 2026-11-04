# Review — FT-0010 EXP-0010C Control Semantics and Residual Error Validation: plan and completion

Reviewer: advisor agent (Opus 5.5), consulted twice by the implementing Claude agent: after orientation and a draft plan (before any code), and after the first full commit (e9772df: results, records, handoff). The briefs asked for independent challenge of attribution method, metric parameters, holdout discipline, provenance and the recommendation.
Date: 2026-11-04
Scope: `analysis/control_semantics/`, `tests/test_control_semantics.py`, `results/control_semantics/`, `docs/experiments/EXP-0010C-control-semantics-validation.md` and the canonical documents.

A review is advice, not an accepted change. Dispositions are at the end.

## Plan review (before implementation)

1. **Attribute false alarms by the triggering detection's own box**, with a fixed priority order and a truncation flag, not by foot proximity; the EXP-0010B humanoid rescoring cleared a zone only when the projected humanoid foot was within 64 px and humanoid geometry was never validated. Include low-score detections; crop-audit only the unexplained bucket.
2. **Policy feasibility depends on class mix:** compute class counts and a humanoid geometry check for the unused layouts first; if no fresh layout has humanoids, A and B coincide; take at most two fresh layouts and keep the rest in reserve.
3. **Derive metric parameters from measurement, not scores:** back-project detection feet to the floor and measure error in metres by distance; handle truncated detections symmetrically; extend the threshold grid below 0.05; keep the same TARGET bar; report the v2 metric as written beside v3 on the fresh holdout.
4. **Preregister the mapping from outcomes to recommendations** inside the protocol before the holdout; do not fine-tune in this objective.
5. **Provenance:** commit the EXP-0010B scratch post-hoc scripts unchanged; write every new diagnostic as committed code; record whole-repository state (commit, porcelain, diff hash) in a new function; the holdout guard requires analysis/ and tests/ clean; commit the lock before downloading holdout video; run long jobs in the foreground one at a time; keep timing short and interleaved; list what is unmeasured.

## Completion review (after e9772df)

1. **The `other_object` attribution bucket was contaminated** (partial humanoid and partial vehicle overlaps fell through to it; 62% on L05, 61% on L08, 25% on L03); L05 has no forklifts, so "vehicle confusion" was unsupported as stated. Split it, rerun to NEW files (fa_attribution.json is hashed in lock.json), add a unit test for the priority order, and crop-audit the buckets on L03 (low-score clutter, vehicle, humanoid mismatch) to separate annotation problems from detector errors.
2. **The 0.75 m margin was measured on persons only but policy B scores humanoids with it;** extend the localization measurement to humanoids and check whether L03 still fails if both position-mismatch buckets were label noise.
3. **Coverage must appear in the headline numbers:** the v3 metric ignores 60% of L08 person rows and about 70% of L05 B agents (people truncated at the frame edge) and excluded 36% of L05's tall detections; "B passes L05" must never appear without that.
4. **Keep the verdict, reframe the recommendation:** CHANGE_WORKLOAD is the plain reading of the locked rule; do not escalate to STOP or soften it; drop "define the monitoring requirement the detector demonstrably serves" (target-tuning) and "near-field cameras" (contradicts the coverage point); the workload change must come from a founder-set requirement and then be validated on a new fresh holdout or turned into a model objective.
5. **Disclosures:** timing on battery is a deviation from protocol_v3 (AC), not just a confound; every wrapped manifest's `command` field names run_cag and is wrong for fresh roles; the label `excluded_truncated_or_short` counts only tall edge-touching detections.
6. **Closeout:** write this file; break the commit-hash circularity between the record and the handoff; push; answer the founder's three-way question with the reframed CHANGE_WORKLOAD.

## Disposition (implementer)

- **Plan review:** all adopted. Class mix and humanoid geometry were surveyed before the policies (and decided that humanoids appear in 33 of 36 unused cameras); margin and cell size are derived from measured localization (not from scores); the grid extends to 0.01; the decision rule was locked in protocol_v3; the EXP-0010B scratch scripts are committed unchanged under `analysis/control_gate/posthoc/`; provenance is a new module and a guard; the lock was committed before the fresh videos were downloaded. **Not adopted as advised:** nothing; the tau rule (macro balanced accuracy) was my choice and is flawed (KI-018), disclosed.
- **Completion 1:** confirmed. Corrected classifier written as `fa_attribution2.py` with a priority-order test (5 tests); rerun to `fa_attribution_v2.json` and `fa_attribution_gp_v2.json` (originals kept); on L08 the old bucket was mostly humanoid-partial (28%) and vehicle-partial (25%) overlaps with only 8% true vehicle overlap, on L05 53% is genuine AMR/transporter confusion (survey: AMRs in every frame). The L03 crop audit shows cardboard boxes on a conveyor (clutter), AMRs/transporters/pallet loads (vehicle bucket) and detections on robots whose box bottom misses the labelled floor position (humanoid bucket); no unlabelled person was seen. If both position-mismatch buckets were label noise L03 would still be at 0.112 false alarms per predicted cell (above 0.10).
- **Completion 2:** confirmed and the margin does not hold for one robot: AgilityDigit ground error median 0.33 m, 95th percentile 1.3-1.6 m, 24-26% beyond 0.75 m (FourierGR1T2 is like persons). Recorded as R2b and a limitation; the locked margin and the verdict are unchanged.
- **Completion 3:** coverage added to R7 and to PROJECT_STATE.
- **Completion 4:** recommendation rewritten as advised (CHANGE_WORKLOAD, founder-set requirement, no tuning of the detector, metric, range or camera selection).
- **Completion 5:** battery recorded as a protocol deviation; wrapper now records the real command (earlier manifests are not rewritten and are described in the record); label corrected in code comments and the record (the JSON key is unchanged to keep outputs comparable).
- **Completion 6:** done; hashes are recorded in the record and the handoff in a final docs commit.

## What the reviewer did not do

The reviewer did not re-execute any experiment and relied on the implementer's transcript and artifacts. Its claim that L05 has no vehicles was checked against `survey.json` by the implementer and was wrong for AMRs/transporters (present in every frame) but right for forklifts/pallet trucks (absent); the corrected split therefore found real AMR confusion on L05.
