# REV-2026-11-07: EXP-0013 Phase 1.1 closeout, hardening round (advisor review of the implementation)

Status: advice from an independent advisor pass over the hardening work; not an accepted architectural change. The advisor saw the full working session (diffs, test output) before the final commits. Implementation notes live in `experiments/EXP-0013-phase1-1-closeout.md` section 15; this file records only what the review said and what was done about it.

## Points raised and dispositions

| # | advisor point | disposition |
|---|---|---|
| 1 | The test-log job was started as an unmanaged background loop; confirm it finished and do not commit its marker file. | Checked for completion by file, marker removed before staging; the logs used for the manifest were produced by a harness-tracked run from a clean committed tree. |
| 2 | Order the commits so the manifest records a clean tree; separate the earlier session's test logs from the new ones. | Order: code commit, logs commit, manifest commit. `tests_observed_this_session` keeps the earlier `chunk*.log` entries and the new `hardening_*.log` entries; they are distinguishable by log filename (stated in the record). |
| 3 | The record still contradicted itself: section 14 still carried the "exact bound" forward and section 6(iii) still read the layout bootstrap as unconditional. **Blocking.** | Both corrected: the bootstrap is described as conditional on the sampled layouts being an i.i.d. draw with spawns nested in layouts (within-pool sensitivity, not a population guarantee); the carry-forward list no longer mentions the bound. Repo-wide grep for the withdrawn figure repeated over `docs/experiments`, `PROJECT_STATE`, `BENCHMARKS`, `OPEN_QUESTIONS`, `ROADMAP`: only historical/superseded mentions remain, each labelled. The older advisor review (`REV-2026-11-07-exp-0013-closeout-advisor-review.md`) is a dated record and is left unedited. |
| 4 | The blackout-spanning-termination fixture stepped v1 but never compared it. | Added the same per-step physics comparison used by every other fixture. |
| 5 | `analyze_state.py` probe default changed from the held-out set to training layouts. | Documented explicitly (record section 15, row 5b) with the instruction to set `PS_PROBE_SPLIT=dev` for a held-out-style probe. Frozen EXP-0012 diagnostics are unaffected. |
| 6 | Prove the raw data are untouched with `git diff --stat`, not by assertion. | Done in the final report (`git diff --stat 0a0a34c -- results/`). |
| 7 | Persist the advisor review as a separate file. | This file. |
| 8 | Merge-readiness cannot be unqualified: the RTL module was not run. | Verdict stated as "ready pending a founder-approved RTL chunk"; nothing under `rtl/` changed. |

## Reviewer's standing caveats (carried into the record)

- The zero-failure result is a descriptive count on an already-examined DEV pool, not evidence about unseen layouts.
- The runner limitation (KI-031) matters only for a v2 environment; it must be designed around, not patched silently, because a changed first-observation rule would alter EXP-0012 behaviour.
- The characterization tests pin current behaviour; if v2 changes the interface, they should be replaced, not loosened.
