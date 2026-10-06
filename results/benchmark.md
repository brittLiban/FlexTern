# FlexTern Phase 1 — Generic Synthesis Comparison

**Type of result: generic synthesis / resource comparison.** Not physical PPA. No FPGA target, no standard-cell library, no timing constraints, no power model. Every number below is a count of technology-independent Yosys cells after `synth -flatten` with the default ABC mapping. See `synth_environment.txt` for exact tool versions and `*_synth.txt` for the raw logs these numbers are copied from.

Experiment record: `docs/experiments/EXP-0001-y-core.md`.

## Headline table (Yosys 0.69, YoWASP build — canonical)

| Metric | Y-Core | INT8 mul baseline | Y-Dot16 | INT8 dot16 baseline |
|---|---:|---:|---:|---:|
| Activation width | 8-bit signed | 8-bit signed | 16 × 8-bit signed | 16 × 8-bit signed |
| Weight representation | ternary, 2-bit encoded | signed INT8 | 16 × ternary, 2-bit | 16 × signed INT8 |
| Output width | 9-bit signed | 16-bit signed | 13-bit signed | 20-bit signed |
| General multipliers inferred (`$mul`) | **0** | 1 | **0** | 16 |
| Coarse cells before mapping | 2 `$eq`, 1 `$neg`, 1 `$pmux` | 1 `$mul` | 32 `$eq`, 16 `$neg`, 16 `$pmux`, 16 `$add` | 16 `$mul`, 16 `$add` |
| Total generic logic cells | **33** | 407 | **1,330** | 8,646 |
| Add/sub-related cells (XOR + XNOR) | 5 | 136 | 341 | 2,991 |
| Mux cells (`$_MUX_`) | 1 | 5 | 20 | 70 |
| Simulation correctness | PASS (768 exhaustive + 256 reserved) | PASS (65,536 exhaustive) | PASS (26 det + 10,000 random + 2,000 Python x-check) | PASS (7 det + 10,000 random) |
| Baseline ÷ FlexTern, total cells | — | **12.3×** | — | **6.5×** |

Source logs: `y_core_synth.txt`, `int8_mul_baseline_synth.txt`, `y_dot16_synth.txt`, `int8_dot16_baseline_synth.txt`.

Full gate breakdown (Yosys 0.69):

| Cell type | Y-Core | INT8 mul | Y-Dot16 | INT8 dot16 |
|---|---:|---:|---:|---:|
| `$_AND_` | 8 | 75 | 184 | 1,185 |
| `$_ANDNOT_` | 2 | 10 | 31 | 117 |
| `$_NAND_` | 9 | 166 | 574 | 4,002 |
| `$_OR_` | 7 | 5 | 134 | 110 |
| `$_ORNOT_` | 1 | 8 | 26 | 139 |
| `$_NOR_` | 0 | 1 | 6 | 14 |
| `$_NOT_` | 0 | 1 | 14 | 18 |
| `$_XOR_` | 0 | 117 | 255 | 2,662 |
| `$_XNOR_` | 5 | 19 | 86 | 329 |
| `$_MUX_` | 1 | 5 | 20 | 70 |
| **Total** | **33** | **407** | **1,330** | **8,646** |

## Cross-version check (Yosys 0.52, native Linux under WSL2)

The same scripts were run with a different Yosys build to see how sensitive the counts are to tool version. Logs in `crosscheck_yosys052/`.

| | Y-Core | INT8 mul | Y-Dot16 | INT8 dot16 |
|---|---:|---:|---:|---:|
| Total generic cells (Yosys 0.52) | 60 | 423 | 1,694 | 8,886 |
| Baseline ÷ FlexTern (Yosys 0.52) | — | 7.1× | — | 5.2× |
| `$mul` inferred | 0 | 1 | 0 | 16 |

The structural conclusion (no multiplier; ternary datapath is several times smaller) holds across both versions. The exact ratio does not: Y-Core alone ranges from 33 to 60 cells depending on tool version, so **single-digit cell counts and precise ratios should not be quoted as stable facts**. The 16-element ratio (5.2× to 6.5×) is more stable than the single-element ratio (7.1× to 12.3×).

## Observations (labeled)

- **MEASURED.** Neither Y-Core nor Y-Dot16 infers a `$mul` cell. The ternary operation reduces to equality decode, a 9-bit negate, and a 3-way select.
- **MEASURED.** At 16 elements, the ratio between baseline and ternary falls from 12.3× to 6.5× (Yosys 0.69). The accumulation adder chain is a cost both designs share.
- **INFERRED.** Subtracting 16 × single-element cost from the dot16 totals gives a rough accumulation cost of ~800 cells for Y-Dot16 (13-bit adds) and ~2,100 for the INT8 baseline (20-bit adds). ABC optimizes across element boundaries, so this split is approximate.
- **INFERRED.** Y-Dot16's accumulation is roughly 60% of its total logic. Any further gains at this level would come from the adder structure, not from the Y-Core itself.

## What this DOES prove

- The Y-Core RTL is functionally correct over its entire valid input domain, including -128 × -1 = +128.
- Y-Dot16 composes 16 Y-Cores correctly, verified by exhaustive-style deterministic cases, 10,000 random vectors, and 2,000 vectors generated independently from the Python reference.
- Ternary multiplication can be expressed in synthesizable RTL with no general multiplier, and Yosys confirms none is inferred.
- Under generic synthesis, the ternary datapath requires several times fewer logic cells than an equivalent-width INT8 multiply datapath, at both single-element and 16-element scale, across two Yosys versions.
- The flow is reproducible from a clean checkout with `scripts/run_all.ps1`.

## What this DOES NOT prove

- **That FlexTern beats NVIDIA GPUs**, or any commercial accelerator. Nothing here models a GPU, DSP slice, tensor core, or any real MAC unit.
- **That FlexTern uses X times less power.** No power model exists in this flow.
- **That FlexTern produces more tokens per second.** No throughput was measured; the designs are combinational and unclocked.
- **That an ASIC will achieve any particular frequency or area.** Generic Yosys cells have no physical meaning. FPGA LUT counts and ASIC gate area would both differ.
- **That a complete transformer is faster.** No memory system, control, interconnect, or activation handling exists.
- **That the ratio is intrinsic.** The INT8 baseline computes a strictly more general function (any INT8 weight) with a wider output (16 vs 9 bits, 20 vs 13 bits). Part of the difference is reduced function, not architectural insight. A fair system comparison would also need to account for ternary models being larger (more parameters) to reach equal quality, which this experiment does not address.
- **That the exact ratio is stable.** The two Yosys versions disagree by almost 2× on Y-Core alone.

## Environment

See `synth_environment.txt` (canonical run) and `crosscheck_yosys052/environment.txt` (cross-check).

Note on runtime: the YoWASP (WebAssembly) Yosys takes roughly 13 minutes on `int8_dot16_baseline` because ABC runs slowly under WASM on the 16-multiplier netlist. Native Yosys does the same step in under 3 seconds. The results are identical in kind; only wall time differs.
