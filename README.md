# FlexTern

FlexTern explores hardware/software co-design for ternary and ultra-low-precision AI. The long-term research thesis is that compute designed around weights restricted to {-1, 0, +1} may achieve materially better efficiency than conventional accelerator architectures. That thesis is **unproven**. This repository is where it gets tested, one controlled experiment at a time.

## Phase 1 — Y-Core v0.1

Y-Core is the smallest experimental ternary compute element. It replaces a multiplier with a three-way selection:

```
weight +1  ->  +activation
weight  0  ->  zero / skip
weight -1  ->  -activation
```

Phase 1 asks one narrow question: can this be implemented correctly in RTL without a multiplier, and how does its generic logic compare to a conventional signed INT8 × INT8 multiplier, at one element (Y-Core) and at sixteen elements (Y-Dot16)?

Phase 1 is **complete**. Results are in [`results/benchmark.md`](results/benchmark.md) and the full experiment record is [`docs/experiments/EXP-0001-y-core.md`](docs/experiments/EXP-0001-y-core.md). Read the "What this does NOT prove" sections before quoting any number.

### Vocabulary

| Term | Meaning here |
|---|---|
| **Activation** | The input value flowing through a neural network layer. In Phase 1, a signed 8-bit integer, -128..+127. |
| **Weight** | A learned parameter multiplied against an activation. Conventionally 8, 16, or 32 bits. |
| **Ternary weight** | A weight restricted to exactly three values: -1, 0, +1. Stored in 2 bits (`00`=0, `01`=+1, `11`=-1, `10`=reserved). |
| **Dot product** | Sum of element-wise products of two vectors. The core operation of matrix multiplication and therefore of inference. |
| **RTL** | Register-transfer level hardware description. The `.sv` files in `rtl/` describe logic, not software. |
| **Simulation** | Running RTL in a simulator (Icarus Verilog) with a testbench that checks outputs against expected values. Proves correctness. |
| **Synthesis** | Translating RTL into a netlist of logic cells (Yosys). Shows what structure the logic requires. Here it is *generic*: no target device, so no area, timing, or power. |

## Phase 2 — Real workload characterization

Phase 2 asked what a real ternary transformer actually demands. The reference workload is `microsoft/bitnet-b1.58-2B-4T` (MIT, pinned revision in ADR-0003). `analysis/ternary_model_profile.py` reads the packed checkpoint directly, counts every one of its 2.08 billion ternary weights, and writes [`results/workload_profile/`](results/workload_profile/). The experiment record is [`docs/experiments/EXP-0002-real-ternary-workload-profile.md`](docs/experiments/EXP-0002-real-ternary-workload-profile.md).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_workload_profile.ps1
```

Downloads each pinned checkpoint into the Hugging Face cache on first run (never into the repo), verifies its SHA-256, then profiles it in under a minute. Needs `pip install numpy huggingface_hub`.

**EXP-0002B** repeated the profile on an independent native ternary family, TII Falcon-E 1B and 3B, with the same tool, to separate what generalizes from what was Microsoft-specific. Comparison: [`results/workload_profile/cross_model/comparison.md`](results/workload_profile/cross_model/comparison.md); record: [`docs/experiments/EXP-0002B-cross-model-validation.md`](docs/experiments/EXP-0002B-cross-model-validation.md).

**EXP-0004** is the first model-side experiment: it quantizes Falcon-E 1B's BF16 output head to INT4 and its KV cache to ~4 bits, and measures quality (lm-eval tasks, teacher-forced incremental perplexity through the real cache), memory, and latency on a local GPU. Runs under WSL2 (`scripts\run_model_precision.ps1 -Wsl`); record: [`docs/experiments/EXP-0004-low-precision-head-kv.md`](docs/experiments/EXP-0004-low-precision-head-kv.md); outputs in [`results/model_precision/`](results/model_precision/).

**EXP-0003** is an analytical compute-vs-memory bottleneck model over all three workloads: where ternary specialization has end-to-end headroom and where the output head, KV cache, or attention cap it. Everything in it is modeled, not measured. Run with `scripts\run_roofline.ps1`; outputs in [`results/roofline/`](results/roofline/); record: [`docs/experiments/EXP-0003-compute-memory-roofline.md`](docs/experiments/EXP-0003-compute-memory-roofline.md).

## Repository layout

```
AGENTS.md          How humans and AI agents work in this repo. Agents read this first.
README.md          This file.
docs/              Canonical project documentation. Start at docs/INDEX.md.
  decisions/       Architecture Decision Records
  experiments/     Experiment records, including negative results
  reviews/         Independent reviewer critiques
  handoffs/        Agent-to-agent handoff notes
analysis/          Workload-characterization tools (EXP-0002); no RTL
reference/         ternary_op.py — Python golden reference the RTL is checked against
rtl/               y_core.sv, y_dot16.sv, int8_mul_baseline.sv, int8_dot16_baseline.sv
tests/             Python unit tests and SystemVerilog testbenches
synth/             Yosys synthesis scripts, one per module
results/           Synthesis logs and benchmark.md (committed, measured values only)
scripts/           run_all.ps1 (full validation flow), gen_vectors.py (cross-check vectors)
build/             Compiled simulations and sim logs (ignored by git)
```

## Running everything on Windows

### Prerequisites

| Tool | Install | Notes |
|---|---|---|
| Python 3.11+ | `winget install --id Python.Python.3.12 --exact` | The Microsoft Store `python.exe` alias in `WindowsApps` is not a real interpreter; the runner skips it. |
| Icarus Verilog 12 | `winget install --id Icarus.Verilog --exact` | Installs to `C:\iverilog\bin`. Add that to PATH, or the runner finds it there anyway. |
| Yosys | `python -m pip install yowasp-yosys` | WebAssembly build that runs inside Python. Chosen because Windows **Smart App Control** blocks the unsigned native `yosys.exe` from OSS CAD Suite. The runner also honors `$env:FLEXTERN_YOSYS`, `yosys` on PATH, or `C:\oss-cad-suite\bin\yosys.exe` if one of those actually runs. WSL with `apt install yosys` is a further fallback. |

Git is assumed present.

### Full validation

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

Simulation only, if Yosys is not installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -SkipSynth
```

The runner stops at the first failure and prints PASS only when the tool exited 0 **and** the testbench printed its success token.

Expect about 14 minutes. Almost all of it is one step, the INT8 dot16 baseline, where ABC runs slowly under the WebAssembly Yosys (see `docs/KNOWN_ISSUES.md` KI-001). Everything else takes seconds.

### Individual steps

```powershell
# Python reference tests
python -m unittest tests.test_reference -v

# Cross-check vectors (Python golden reference -> hex file read by the Y-Dot16 testbench)
python scripts\gen_vectors.py

# Y-Core exhaustive (768 valid cases + 256 reserved-encoding cases)
iverilog -g2012 -Wall -o build\y_core_tb.vvp rtl\y_core.sv tests\y_core_tb.sv
vvp build\y_core_tb.vvp

# Y-Dot16 (deterministic + 10,000 random + 2,000 Python cross-check)
iverilog -g2012 -Wall -o build\y_dot16_tb.vvp rtl\y_core.sv rtl\y_dot16.sv tests\y_dot16_tb.sv
vvp build\y_dot16_tb.vvp

# INT8 baselines (65,536 exhaustive products + dot16 tests)
iverilog -g2012 -Wall -o build\int8_baseline_tb.vvp rtl\int8_mul_baseline.sv rtl\int8_dot16_baseline.sv tests\int8_baseline_tb.sv
vvp build\int8_baseline_tb.vvp

# Generic synthesis (logs land in results\). yowasp-yosys lives in Python's Scripts\ folder.
yowasp-yosys -q synth\y_core.ys
yowasp-yosys -q synth\y_dot16.ys
yowasp-yosys -q synth\int8_mul_baseline.ys
yowasp-yosys -q synth\int8_dot16_baseline.ys
```

### Phase 1 result in one line

Under generic synthesis (Yosys 0.69), Y-Core needs 33 logic cells versus 407 for a signed INT8 × INT8 multiplier, and Y-Dot16 needs 1,330 versus 8,646 for the INT8 equivalent, with no multiplier inferred anywhere in the FlexTern modules. A second Yosys version gives smaller ratios (7.1× and 5.2×). These are structure counts, not area, speed, or power. See [`results/benchmark.md`](results/benchmark.md) for what this does and does not prove.

## Current limitations

- **Generic synthesis only.** Cell counts come from Yosys' internal library with no target device. They measure logic structure, not area, frequency, or power on any real FPGA or ASIC.
- **The INT8 baseline is a generic RTL multiplier**, not a model of any commercial accelerator's MAC, DSP slice, or tensor core. It also computes a strictly more general function than Y-Core, so part of any difference is reduced function rather than architectural insight.
- **Combinational only.** No clocks, no pipelining, no timing analysis.
- **No memory system.** Weight storage, fetch, and unpacking are outside Phase 1, and may dominate at system scale.
- **No neural network.** Nothing here runs a model.
- **No GPU comparison** of any kind.

## Planned direction

```
Y-Core
  ->
Y-Dot16
  ->
Y-Array
  ->
ternary matrix engine
  ->
FPGA
  ->
real ternary neural-network layer
```

This is direction, not a schedule. The next step is chosen by the founder after reviewing Phase 1 evidence. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/CURRENT_WORK.md`](docs/CURRENT_WORK.md).

## Documentation

Start at [`docs/INDEX.md`](docs/INDEX.md). Agents start at [`AGENTS.md`](AGENTS.md).

## Research disclaimer

FlexTern is research. Nothing here is a product. No performance, efficiency, or cost advantage over any real system has been demonstrated. Claims in the documentation are labeled known, measured, inferred, hypothesized, target, or speculative. An unlabeled performance statement should be treated as unverified.
