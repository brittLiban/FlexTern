# tests/

Simulation testbenches and reference models. Every design in `rtl/` should have a test here that proves functional correctness before it is synthesized or measured.

## Python test tiers (EXP-0013 Phase 1.1 closeout)

`tests/test_*.py` are unit-test modules (they are not the simulation data split named `"test"`; see `analysis/predictive_state/split_manifest.json`). Run them through `scripts/run_tests.py`, which builds chunks from recorded per-module durations (`tests/test_budget.json`) so that no single invocation exceeds the 15-minute budget:

| tier | command | contents | needs |
|---|---|---|---|
| smoke | `python scripts/run_tests.py smoke` | reserved-split guard, paired-replay env, OccPick2D-LA env | numpy |
| benchmark | `python scripts/run_tests.py benchmark` | smoke + observation-matched controllers | numpy |
| predictive | `python scripts/run_tests.py predictive` | benchmark + EXP-0012 env/model tests + gated-encode / executed-work regression | torch |
| full | `python scripts/run_tests.py full` | every module in <= 900 s chunks (`full --chunk K`); the 21 non-RTL modules take about 100 s in total; `test_hardware_gate` (iverilog RTL simulations, about 29 min, INFERRED; it timed out at 900 s in the closeout run) is its own chunk and needs `--approved` | torch; iverilog for the RTL module |

Do not run `full` on every research iteration; it is a merge-readiness check. Durations are in the closeout record; `python scripts/run_tests.py list` prints the current estimates.
