# analysis/

Workload-characterization tooling. Phase 2 (EXP-0002) analyzes a real ternary transformer checkpoint to decide what hardware problem FlexTern should attack next. No RTL here.

| File | Purpose |
|---|---|
| `ternary_model_profile.py` | Main tool. Downloads (pinned revision) or opens a local `microsoft/bitnet-b1.58-2B-4T` checkpoint, verifies file integrity and byte-level tensor coverage, unpacks every 2-bit ternary tensor, counts -1/0/+1, measures zero structure, inventories shapes/ops, computes storage, traffic, and accumulator-width tables, and writes `results/workload_profile/`. |
| `safetensors_reader.py` | Minimal safetensors header parser + memmap reader, BF16 decode, and the BitNet 2-bit unpack/pack routines transcribed from `transformers/integrations/bitnet.py`. |

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_workload_profile.ps1
```

or directly:

```powershell
python -m unittest tests.test_analysis -v
python analysis\ternary_model_profile.py
```

Requires `numpy` and `huggingface_hub` (`pip install numpy huggingface_hub`). The checkpoint (1.18 GB) lands in the Hugging Face cache outside the repository and is never committed. Use `--local-file` / `--local-config` to point at an existing copy.

## Outputs (`results/workload_profile/`)

- `workload_summary.md` — human-readable tables, generated, do not hand-edit
- `model_summary.json` — config, provenance, integrity checks, global distribution, parameter accounting
- `weight_distribution.csv` — one row per ternary tensor (210 rows)
- `weight_distribution_by_module.csv`, `weight_distribution_by_layer.csv`
- `layer_shapes.csv` — linear-operation inventory incl. the tied BF16 head
- `dot_length_summary.csv`, `expected_terms_per_dot.csv`
- `operation_inventory.csv` — per-block ternary and non-ternary operations with static op counts
- `memory_representation.csv`, `decode_traffic_model.csv`, `accumulator_widths.csv`
- `environment.txt` — versions, revision, SHA-256, command, runtime

All arithmetic/traffic numbers are INFERRED static estimates; weight counts and shapes are MEASURED.
