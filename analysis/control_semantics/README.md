# EXP-0010C Control Semantics and Residual Error Validation

Record: `docs/experiments/EXP-0010C-control-semantics-validation.md`. Protocol: `protocol_v3.json`, locked by `results/control_semantics/lock.json`.
Builds on `analysis/control_gate/` (EXP-0010B) and `analysis/edge_inference/` (Stage A); neither is modified.

| file | role |
|---|---|
| `provenance.py` | whole-repository git state for manifests; `require_clean_code()` guard (analysis/ and tests/) |
| `ground.py`, `gp_metric.py` | floor-plane geometry; ground-plane zone-occupancy metric for the two policies |
| `survey.py`, `select_fresh.py` | GT-only class-mix / geometry survey; fresh-holdout layout rule |
| `fa_attribution.py`, `localization.py`, `miss_attribution.py` | post-hoc diagnostics on development data |
| `dev_eval.py` | development scoring and per-policy tau (macro balanced accuracy; see KI-018 for its flaw) |
| `run_cs.py` | CLI: `lock`, `fetch`, `detect`, `score`, `replay` (fresh-holdout commands refuse without a committed, matching lock and clean code) |
| `fa_attribution_gp.py`, `tau_sweep_posthoc.py`, `timing_summary_cs.py` | post-hoc analyses run after the one fresh scoring |

Run inside the WSL venv from the repo root (`FLEXTERN_EDGE_CACHE=~/flextern_edge`): `survey`, `select_fresh`, `localization`, `fa_attribution`, `miss_attribution`, `dev_eval`; commit; `run_cs lock`; commit; `run_cs fetch`; `run_cs detect YT <hold_L05|hold_L03>`; `run_cs score <role>`. Tests: `python -m unittest tests.test_control_semantics`.
