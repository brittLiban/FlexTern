# FT-0010 Stage A — continuous edge-inference baseline harness

Protocol and results: `docs/experiments/EXP-0010-edge-inference-baseline.md`. Frozen protocol: `protocol.json`.
Generated results: `results/edge_inference/` (`summary.md` is the readable roll-up).

## What is here

| file | role |
|---|---|
| `protocol.json` | frozen splits, scoring policy, streaming policy, tolerances (written before any model run) |
| `dataset_audit.py` | HF metadata audit; groups scenes into **layout groups** (identical camera geometry or `map.png`) |
| `gt.py` | streams the 15–280 MB ground-truth JSON into per-camera compact arrays; accepts both key styles |
| `pipeline.py` | shared numpy/OpenCV preprocessing + person-only postprocessing (same code in front of both runtimes) |
| `models.py` | official torchvision weights, network-only ONNX export, `TorchRunner` (B0), `OrtRunner` (B1) |
| `evaluate.py` | COCO-style matching with ignore regions, AP50, P/R/F1, block bootstrap |
| `replay.py` | paced streaming harness (feeder + bounded drop-oldest queue + worker), offline runner, trace invariants |
| `run_stage_a.py` | CLI orchestrator (`prepare`, `parity`, `offline`, `calibrate`, `replay`, `align`) |
| `report.py` | aggregates run summaries into `results/edge_inference/summary.md` |
| `alignment_check.py` | draws GT on decoded frames for a visual alignment check |
| `ssdlite320_anchors.npy` | the 3234 default boxes of the architecture at 320x320 (derived from the official model; hash in `identity.json`) |

## Environment used for the recorded runs

WSL2 Ubuntu venv (`~/ftvenv`, Python 3.14): `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128` (the PyPI torchvision wheel
does **not** match this torch build — its C++ ops fail to register; install the cu128 wheel with `--no-deps`),
`onnx`, `onnxruntime`, `onnxscript`, `av`, `opencv-python-headless`, `ijson`, `psutil`.
Host Python (Windows) cannot import torch (KI-005); the numpy-only tests run there, the rest in WSL.

## Reproduce (from the repo root, inside WSL)

```bash
export FLEXTERN_EDGE_CACHE=~/flextern_edge          # raw videos/GT live here, never in git
python -m analysis.edge_inference.dataset_audit      # layout groups (small downloads)
# download the three clips + their ground truth (see EXP-0010 "Data"), then:
python -m analysis.edge_inference.run_stage_a prepare
python -m analysis.edge_inference.run_stage_a parity
python -m analysis.edge_inference.run_stage_a offline calibration B0
python -m analysis.edge_inference.run_stage_a calibrate     # fixes tau; unlocks the holdout
python -m analysis.edge_inference.run_stage_a offline development B0 ; ... B1
python -m analysis.edge_inference.run_stage_a align
python -m analysis.edge_inference.run_stage_a offline holdout B0 ; ... B1
python -m analysis.edge_inference.run_stage_a replay holdout B1 --rep 0   # 5 min each, real time
python -m analysis.edge_inference.report
python -m unittest tests.test_edge_inference -v
```

Raw per-detection files (score >= 0.001, ~tens of MB) are kept in `$FLEXTERN_EDGE_CACHE/runs/` and referenced by SHA-256 in
each run's `summary.json`; the committed bundle contains manifests, summaries and per-frame timing traces.
