# Advisor review: EXP-0013 Phase 1.1 closeout (two consultations, 2026-11-07)

Status: advice, not an accepted change (AGENTS.md §3). Reviewer: stronger-model advisor with the full session transcript. Dispositions are by the implementation agent.

## Consultation 1 (before substantive work)

| advice | disposition |
|---|---|
| Reproduce the historical v1 run with a passive `valid` logger before changing anything; confirm per-episode identity with `paired.json`; measure the defect; check which controllers read `valid`. | Done (15/15 rows identical; privileged controllers never read `valid`, so only A-vs-B style comparisons were compromised by schedule differences). |
| Subclass `LAEnv`; inject nothing into `env.py`; blackout uniforms/durations keyed by (episode, t); full-frame noise field per (episode, t); assert `n_picks == 1`; mind that `observe()` runs inside `BatchEnv.step`; add a transition-equivalence test against v1 and a v1 negative-control test. | Done, via a guarded RNG stand-in so `step`/`render` code is inherited unchanged. |
| Batch-size/order independence tests; stand-still vs random vs drive controllers. | Done (`tests/test_env_paired.py`, 10). |
| Cluster bootstrap over unique layouts; exact one-sided bound when there are zero failures; report "0 discordant pairs" instead of p = 1.0. | Done (`closeout-stats-1`). |
| TEST audit: timing scripts use the EXP-0012 env; separate EXP-0012's legitimate TEST scoring; static guard with allow-list; keep a flag for the historical trace. | Done (`split_manifest.json`, `tests/test_split_guard.py`, `FLEXTERN_ALLOW_TEST_SPLIT`). |
| Run torch tests, capture AC power before/after timing, check the checkpoint exists, verify EXP-0012 merge status with `git merge-base`, update the founder-style memory, do not run the full suite. | Done (EXP-0012 is an ancestor of `main`). |

## Consultation 2 (after the first commit, before the final record)

| finding | disposition |
|---|---|
| The v1 defect count compared whole runs including steps after termination; the live-window count is the relevant one. | Fixed: headline is now 114-130/200 (LA_NOM) and 177-192/200 (LA_OCC); whole-run counts kept as a second figure. Commit `417601f`'s message and its first version of the record quoted the larger whole-run counts (163-193 / 195-200); the second commit supersedes them. |
| Raw `repo_sha` was the base commit while the producing code was uncommitted. | Re-ran LA_OCC from the clean committed tree (`417601f`): all compared fields identical (`determinism_rerun_LA_OCC.json`). Raw files record the base commit; the producing code is committed unchanged in `417601f`. |
| "No claim depends on TEST traces" overstated; `tests/test_predictive_state.py` omitted; `test_paired_episodes_identical_across_controllers` is not a pairing test. | Fixed in the record, manifest, KI-030, Phase 1.1 note; docstring added. |
| `run_tests.py` did not refuse over-budget `full` chunks and its timeout could not be lifted with `--approved`. | Fixed. |
| Closed-form schedule test never exercised termination; tables were "produced by" rather than "transcribed from" the script. | Fixed (`t_limit=30`, wording; `analyze_closeout` re-run on the committed tree leaves tables unchanged). |
| Provenance: copy test logs, build the manifest, list out-of-repo side effects, never push to `main`. | Done; see the final report. |
