# FT-0010 Control Adequacy Gate (EXP-0010B)

Record: `docs/experiments/EXP-0010B-control-adequacy-gate.md`. Protocol: `protocol_v2.json` (+ `protocol_v2_amendment1.json`). Results: `results/control_gate/`.
Builds on `analysis/edge_inference/` (Stage A harness, unchanged).

| file | role |
|---|---|
| `gt3d.py`, `consistency.py` | 3D boxes + calibration -> foot points / expected whole-person height; scene reliability rule |
| `zones.py`, `cag_eval.py` | zone-occupancy (primary), presence, count, AP, visibility metrics; block bootstrap |
| `detectors.py` | candidates S0/S1/S2 (SSDLite) and YT/YS (YOLOX) as harness `Engine`s |
| `parity.py` | export of SSD variants and parity vs official torchvision / YOLOX reference code |
| `select_holdout.py` | fresh-holdout layout rule applied to ground truth only |
| `run_cag.py` | CLI: `detect`, `devsel`, `replay`, `feasibility`, `select` (unlocks holdout), `holdout` |
| `fp_crops.py`, `timing_summary.py` | vehicle false-positive audit montage; timing roll-up |

Run inside the WSL venv from the repo root with `FLEXTERN_EDGE_CACHE=~/flextern_edge`; order: `parity`, `select_holdout`, `run_cag detect <cfg> <dev role> --stride 3`, `run_cag devsel`, `run_cag feasibility`, `run_cag select`, `run_cag detect <cfg> hold_* --stride 1`, `run_cag holdout hold_*`. The holdout commands refuse to run until `results/control_gate/dev_selection.json` exists. The YOLOX ONNX files come from the Megvii-BaseDetection release 0.1.1rc0 (Apache-2.0) into `$FLEXTERN_EDGE_CACHE/models/yolox/`; hashes are in each run manifest.
Tests: `python -m unittest tests.test_control_gate`.
