# analysis/

Workload-characterization tooling. Phase 2 (EXP-0002, EXP-0002B) analyzes real ternary transformer checkpoints to decide what hardware problem FlexTern should attack next. No RTL here.

| File | Purpose |
|---|---|
| `model_specs.py` | One entry per supported checkpoint: id, pinned revision, expected size/SHA-256, license, reference-code pointers, scale convention (`multiply` = stored β, `divide` = stored 1/β), per-layer norm names, tied/untied head, latent repo for cross-checks, output directory. The profiler has no model constants of its own. |
| `ternary_model_profile.py` | Main tool. Downloads (pinned) or opens a packed BitNet-format checkpoint, verifies size/SHA-256 and byte-level tensor coverage, unpacks every 2-bit ternary tensor, counts -1/0/+1, measures zero structure and dead channels, inventories shapes/ops, computes storage, traffic, and accumulator-width tables, and writes the model's output directory. |
| `safetensors_reader.py` | Minimal safetensors header parser + memmap reader, BF16 decode, and the BitNet 2-bit unpack/pack routines transcribed from `transformers/integrations/bitnet.py`. |
| `ternarization_crosscheck.py` | HTTP-range fetches a few BF16 latent tensors, applies absmean ternarization (float32 and BF16-emulated), compares with the packed codes, and checks the stored scale against the spec's convention. |
| `cross_model_compare.py` | Joins the per-model outputs into `results/workload_profile/cross_model/comparison.{md,json}`. |
| `roofline_models.py` | EXP-0003 library: per-token stage model (ternary linears, head, attention+KV, other), overlap bounds, Amdahl ceilings, analytical crossovers and break-even equations, residency policies, tiles, accumulator widths, prefill. ANALYTICAL only. |
| `flextern_roofline.py` | EXP-0003 driver: loads the three canonical `model_summary.json` files, verifies invariants, runs the sweeps, writes `results/roofline/`. Run via `scripts\run_roofline.ps1`. |

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_workload_profile.ps1            # all three models
powershell -ExecutionPolicy Bypass -File scripts\run_workload_profile.ps1 -Models falcon-e-1b
```

or directly:

```powershell
python -m unittest tests.test_analysis tests.test_analysis_cross_model -v
python analysis\ternary_model_profile.py --model bitnet-2b4t      # EXP-0002 canonical; writes results/workload_profile/
python analysis\ternary_model_profile.py --model falcon-e-1b      # writes results/workload_profile/falcon-e-1b/
python analysis\ternary_model_profile.py --model falcon-e-3b
python analysis\ternarization_crosscheck.py --model falcon-e-1b
python analysis\cross_model_compare.py
```

Requires `numpy` and `huggingface_hub`. Checkpoints (1.18 GB, 0.67 GB, 1.00 GB) land in the Hugging Face cache outside the repository and are never committed. `--local-file` / `--local-config` point at an existing copy.

## Outputs (per model directory)

- `workload_summary.md` — human-readable tables, generated, do not hand-edit
- `model_summary.json` — config, provenance, integrity checks, distribution, parameters, head, dot lengths with divisibility
- `weight_distribution.csv` (one row per ternary tensor), `weight_distribution_by_module.csv`, `weight_distribution_by_layer.csv`, `dead_channels_by_layer.csv`
- `layer_shapes.csv`, `dot_length_summary.csv`, `expected_terms_per_dot.csv`, `operation_inventory.csv`
- `memory_representation.csv`, `decode_traffic_model.csv`, `accumulator_widths.csv`
- `ternarization_crosscheck.json` (where a latent repo is declared), `environment.txt`

Weight counts and shapes are MEASURED; arithmetic/traffic numbers are INFERRED static estimates.
