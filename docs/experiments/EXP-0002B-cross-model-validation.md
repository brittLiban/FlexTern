# EXP-0002B — Independent ternary workload validation (TII Falcon-E 1B/3B vs BitNet b1.58 2B4T)

Status: Complete
Date: 2026-10-06
Commit: 73dc205 ("research: validate FlexTern workload assumptions across Falcon-E"). Results in `results/workload_profile/` were generated from exactly that tree before the commit existed.

Primary data: `results/workload_profile/cross_model/comparison.md` (generated) and the per-model directories `results/workload_profile/falcon-e-1b/`, `falcon-e-3b/`, and the unchanged EXP-0002 outputs in `results/workload_profile/`. This record interprets them. Labels as in `AGENTS.md` §7.

## Research Question

Which hardware-relevant workload properties measured on Microsoft BitNet b1.58 2B4T in EXP-0002 are model-specific, and which hold on an independent native ternary family well enough to inform the first Youssouf Architecture?

## Hypothesis

Not pass/fail. Expectation going in: the ternary value set, 2-bit storage, per-tensor scale, and INT8 activations would generalize (they are the BitNet b1.58 recipe); dimensions, sparsity structure, and head dominance might not.

## Why It Matters

Every EXP-0002 conclusion came from one checkpoint. Hardening tiles, accumulator widths, sparsity handling, or memory assumptions around it risks overfitting to Microsoft's choices.

## Setup

**Models (ADR-0004).** All packed `main` branches, pinned, SHA-256 verified against Hugging Face etags, never committed:

| | BitNet b1.58 2B4T | Falcon-E-1B-Base | Falcon-E-3B-Base |
|---|---|---|---|
| Repo / revision | `microsoft/bitnet-b1.58-2B-4T` @ `04c3b9ad` | `tiiuae/Falcon-E-1B-Base` @ `f4001b8b` | `tiiuae/Falcon-E-3B-Base` @ `ad18b071` |
| License | MIT | Falcon-LLM License (TII), Apache-2.0-based + AUP | same |
| File | 1,178,623,988 B, 542 tensors | 665,041,488 B, 387 tensors | 998,568,704 B, 515 tensors |
| Config source | `config.json` at revision | same | same |
| Reference code | transformers `integrations/bitnet.py` @ `d6abd133`, `AutoBitLinear` offline | transformers `BitLinear` (no `linear_class` in config) + `tiiuae/onebitllms` @ `9be8e4f3` | same as 1B |
| Native ternary | Model card: trained from scratch with the quantization scheme, 4T tokens | Blog: single pre-training run yields the BitNet model; ~1.5T tokens; TII Triton kernels for weight/activation quant | same |
| Weights stored ternary | Yes, U8 packed 2-bit | Yes, U8 packed 2-bit; `config.quantization_config.modules_to_not_convert = ["lm_head"]` keeps the untied head non-ternary by config | Yes; same config |
| Parameter count | 2,412,820,690 from shapes | 1,719,765,160 from shapes; the model card says "1.8B". Source of the difference not determined (possibly rounding or a different counting convention) | 3,053,586,656 from shapes; card says "3B" |
| Latent checkpoint | `…-2B-4T-bf16` @ `27668139` (fine-tuning copy) | branch `prequantized` @ `ff68bcdd` (input of `quantize_to_1bit`); branch `bfloat16` is a *different non-ternary model* and was not used | branch `prequantized` (sharded; not used) |

**Tooling.** The Phase 2 profiler was refactored to a per-model spec table (`analysis/model_specs.py`); the analysis code contains no model constants. The cross-check script gained BF16-arithmetic emulation. `analysis/cross_model_compare.py` joins the outputs.

## Hardware / Software Environment

As EXP-0002: Windows 11, Python 3.12.10, numpy 2.5.3, huggingface_hub 2.1.1, no torch. Runtimes from cache: 44 s (BitNet), 35 s (1B), 54 s (3B). Deterministic.

## Baseline

The EXP-0002 BitNet results, unchanged.

## Method

1. Provenance from the Hugging Face API (revisions, branches, gating, license tags), header-only HTTP range reads of every branch's `model.safetensors` (dtypes, shapes, tensor counts, file sizes), the model cards, the Falcon-Edge blog, and the pinned reference code. The transformers dispatch was read to determine which linear class each config selects, and `onebitllms` was read for the training kernels and `quantize_to_1bit`.
2. Profiler generalization: spec-driven classification (norm list, tied flag), reconciliation asserts generalized (tensor count = embed + head-if-untied + final norm + L × (7 + 7 + norms); packed rows × 4 and columns = config N and K using `head_dim`; ternary total = shape-derived expectation; packed bytes = ternary/4), head bytes taken from the actual head tensor, op inventory driven by config/spec, KV traffic reported at 4096 and at each model's maximum context, divisibility columns added.
3. Regression gate: BitNet re-profiled offline with the refactored tool; every numeric value in the committed CSV/JSON outputs compared programmatically: **10,929 values, 0 differences**. Only additions (divisibility columns, max-context KV columns, extra JSON fields) and two renamed JSON keys. `tests/test_analysis_cross_model.py` pins the committed BitNet totals.
4. Falcon-E 1B and 3B fully profiled with the same tool. Cross-check on 1B: three `prequantized` tensors fetched by range, absmean-ternarized in float32 and with BF16 emulation, compared to the packed codes; stored scale compared to `1/mean|W|`.
5. Tests: 37 (16 original + 21 new) covering specs, classification for all three layouts, stray-tensor detection, projection dims with `head_dim`, ternary totals, divisibility, accumulator widths at every observed K, and regression pins.

## Metrics

As EXP-0002, per model, plus the cross-model union of K, divisibility, and accumulator widths.

## Results

### Provenance and format (MEASURED)

- Both families store weights as literal ternary codes, 2 bits each, code = q + 1, four per byte **along the output dimension**, one BF16 `weight_scale` per tensor. Reserved code 3: 0 occurrences in all 2.08 + 1.59 + 2.92 billion weights.
- **Scale convention differs.** BitNet (`AutoBitLinear`, offline) stores β = mean|W| and multiplies; Falcon-E (`BitLinear`) stores 1/β and divides. Confirmed on latents: Falcon 1B stored scales 17.375, 8.625, 13.1875 against 1/β = 17.390, 8.637, 13.176 (BF16 precision). Effective weight is β·W_q in both; hardware sees one scalar multiply after accumulation in both.
- **Ternarization reproduced exactly for Falcon.** With BF16 emulation (mean, reciprocal, product each rounded to BF16, as a `torch.bfloat16`-loaded model computes), agreement is **100.000%** on all three tensors (q_proj L0, k_proj L0, down_proj L12). Float32 math gives 99.83 / 99.85 / 100.00%. The packed branch is therefore `quantize_to_1bit(prequantized)` bit for bit. BitNet's gap (98.0–99.4%, worse under emulation) remains a snapshot difference (KI-004).
- Rounding: TII's training kernels use floor(x + 0.5) (half-up); `quantize_to_1bit` and transformers use torch round (half-even). Ranges are unaffected.

### Architecture (MEASURED from config)

| | BitNet 2B4T | Falcon-E 1B | Falcon-E 3B |
|---|---:|---:|---:|
| Total / ternary params | 2.41 B / 2.08 B (86.4%) | 1.72 B / 1.59 B (92.2%) | 3.05 B / 2.92 B (95.6%) |
| Layers | 30 | 24 | 32 |
| Hidden / intermediate (ratio) | 2560 / 6912 (2.70) | 2048 / 9216 (4.50) | 2048 / 13312 (6.50) |
| Heads / KV heads / head dim | 20 / 5 (4:1) / 128 | 16 / 2 (8:1) / 128 | 16 / 2 (8:1) / 128 |
| Context | 4,096 | 32,768 | 32,768 |
| Vocabulary | 128,256 | 32,768 | 32,768 |
| MLP activation / norms per layer | ReLU² / 4 (with sub-norms) | SiLU / 2 | SiLU / 2 |
| Embeddings | tied | untied | untied |
| MLP share of ternary weights | 76.4% | 85.7% | 89.7% |

### Dot lengths (MEASURED)

| Model | K | Modules | % of ternary | ÷64 | ÷128 | ÷256 | ÷512 |
|---|---:|---|---:|---|---|---|---|
| BitNet | 2560 | q,k,v,o,gate,up | 74.5 | yes | yes | yes | yes |
| BitNet | 6912 | down | 25.5 | yes | yes | yes | **no** |
| Falcon 1B | 2048 | q,k,v,o,gate,up | 71.4 | yes | yes | yes | yes |
| Falcon 1B | 9216 | down | 28.6 | yes | yes | yes | yes |
| Falcon 3B | 2048 | q,k,v,o,gate,up | 70.1 | yes | yes | yes | yes |
| Falcon 3B | 13312 | down | 29.9 | yes | yes | yes | yes |

Every model has exactly two K: the hidden size (70–75% of weights) and the intermediate size (25–30%). GCD of the five observed K is 256; 6912 blocks 512. Both observed hidden sizes (2048, 2560) are multiples of 512; it is only BitNet's intermediate size that constrains the common granularity to 256. MEASURED: all observed K are divisible by 256. INFERRED: 256 may be a useful tile granularity. NOT PROVEN: that 256 is optimal, or that unobserved models comply.

### Ternary distribution (MEASURED)

| | BitNet | Falcon 1B | Falcon 3B |
|---|---:|---:|---:|
| % -1 / 0 / +1 | 28.894 / 42.211 / 28.895 | 29.423 / 41.071 / 29.506 | 30.139 / 39.692 / 30.169 |
| \|%+1 − %-1\| | 0.001 pt | 0.083 pt | 0.030 pt |
| Per-tensor zero min / median / max | 36.1 / 41.2 / 60.1 | 31.4 / 40.7 / 59.4 | 32.6 / 40.2 / 58.7 |
| Zero by module, q / k / v / o | 46.8 / 44.7 / 37.6 / 41.8 | 44.5 / 45.4 / 41.1 / 40.3 | 45.0 / 45.5 / 41.6 / 40.5 |
| Zero by module, gate / up / down | 42.4 / 41.6 / 41.3 | 40.9 / 40.2 / 41.4 | 39.5 / 38.8 / 39.7 |

Depth profile differs: BitNet peaks at layer 1 (56.2%) then sits at ~41%; Falcon rises monotonically from ~34% at layer 0 to ~45% around layers 16–22 (1B) and ~42% (3B). Zero rate falls slightly with model scale in Falcon (41.1% → 39.7%).

### Sparsity structure (MEASURED)

| | BitNet | Falcon 1B | Falcon 3B |
|---|---:|---:|---:|
| All-zero rows, all tensors | 19,847 | **0** | **0** |
| All-zero columns, all tensors | 11,241 | 168 | 280 |
| Dead MLP intermediate channels | 9,870 (4.76%; 43.6% in layer 1) | 0 | 0 |

Aligned-group all-zero rates along K match the independence prediction p^g in all three models once all-zero rows are removed; the residual excess in q/k comes from rows with individually extreme zero rates, not from blocks. Falcon's few all-zero columns are hidden dimensions that a v_proj never reads in a given layer, which is of no hardware use.

### Output head (MEASURED; traffic INFERRED)

| | BitNet | Falcon 1B | Falcon 3B |
|---|---:|---:|---:|
| Head tensor | embed (tied), BF16 | `lm_head` (untied), BF16 | `lm_head` (untied), BF16 |
| Head params / bytes | 328 M / 657 MB | 67 M / 134 MB | 67 M / 134 MB |
| Head % of file bytes | 55.7% | 20.2% | 13.4% |
| Head % of linear MACs/token | 13.6% | 4.1% | 2.2% |
| 2-bit ternary bytes/token | 521 MB | 396 MB | 730 MB |
| Head % of stream-once weight bytes | 56% | 25% | 16% |
| KV read/token @4096 / @max ctx | 315 MB / 315 MB | 101 MB / 805 MB | 134 MB / 1,074 MB |

The head is BF16 and non-ternary in both families. Its dominance in BitNet is a 128k-vocabulary effect. In Falcon-E at 32k context, the KV-cache read exceeds the whole weight stream.

### Accumulator widths (INFERRED mathematically; formula brute-force validated at K = 1, 2, 3)

| K | Models | Format [-128,127] bits | Model [-127,127] bits | Max nonzero/row observed |
|---:|---|---:|---:|---:|
| 2048 | Falcon 1B, 3B | 20 | **19** | 2,039 |
| 2560 | BitNet | 20 | 20 | 2,560 |
| 6912 | BitNet | 21 | 21 | 6,785 |
| 9216 | Falcon 1B | 22 | 22 | 9,179 |
| 13312 | Falcon 3B | 22 | 22 | 13,206 |

K = 2048 is the first full-length case where "-128 unreachable" saves a bit (128 × 2048 = 2^18 exactly). Cross-model range: 20–22 bits (format), 19–22 (model). Tiles: 256 → 17/16, 512 → 18/17, 1024 → 19/18.

### Activations (KNOWN from reference code, both families)

Per-token symmetric INT8, dynamic absmax, scale = 127/absmax, clamp [-128, 127], dequantize by the same scale; -128 unreachable, effective range [-127, 127]. Four quantization points per layer (q/k/v share one input; o; gate/up share; down). Independence of this evidence: inference paths for both families run transformers' quantizer by construction; the independent confirmation is TII's own Triton training kernel (`127/row_absmax`, `floor(x·scale + 0.5)`, clamp), which produces the same reachable range by a different rounding rule.

## Interpretation

### Cross-model invariants (two families; evidence, not proof)

Independence caveat: Falcon-E is independent of BitNet in organization, training run, data, architecture, and training kernels, but **not in quantization recipe**. TII explicitly adopted the BitNet b1.58 recipe (absmean ternary weights, INT8 per-token absmax activations). The recipe-level invariants below are therefore "one recipe reproduced by two labs," not independent convergence, and the label for them reads *among BitNet-recipe models*.

**LIKELY GENERALIZING** (recipe-level items: among BitNet-recipe models)
- Ternary weights {-1, 0, +1}; absmean ternarization. *(recipe-level)*
- 2-bit packed storage, code = q + 1, four per byte along the output dimension (shared transformers format; generalizes among transformers-packed checkpoints).
- One scale per tensor, applied after accumulation, outside the dot product; scale cost negligible (≤ 0.03% of ops). *(recipe-level)*
- INT8 per-token absmax activations with -128 unreachable. *(recipe-level; see Q-021)*
- Balanced ±1 (within 0.1 pt).
- ~40% zeros (39.7–42.2%), unstructured along K.
- Exactly two dot lengths per model: hidden and intermediate; both multiples of 256.
- q/k projections sparser than v/o.
- Output head stays BF16 and non-ternary.
- head_dim = 128.

**MODEL-SPECIFIC**
- Exact K values (2048, 2560, 6912, 9216, 13312) and therefore exact accumulator widths (19–22 bits).
- Dead MLP channels and all-zero rows (BitNet only).
- Tied vs untied embeddings; head share of traffic (16–56%, vocabulary-driven).
- Stored scale convention (β vs 1/β).
- MLP activation (ReLU² vs SiLU), sub-norms, intermediate/hidden ratio (2.7–6.5), GQA ratio (4:1 vs 8:1), context (4k vs 32k), depth profile of sparsity.
- Which non-ternary stream dominates bytes: head (BitNet) vs KV cache at long context (Falcon).

**UNCERTAIN**
- Whether K stays a multiple of 256 in future models (five values observed).
- Whether INT8 activations persist (Q-021).
- Whether ~40% zeros is a property of absmean training generally or of these two recipes.
- Whether 2048/2560-class hidden sizes persist at larger scales.

### Impact on Y-Core / Y-Architecture

1. **Survive both families:** signed INT8 activation; ternary 2-bit weight; 9-bit result correct for the format (+128 case unreachable in both); scale outside the primitive.
2. **Clearly Microsoft-specific:** K = 2560/6912; 13/20/21-bit accumulator specifics; dead channels; tied head; head dominance; 00/01/11 code assignment matching no checkpoint (both families use 00/01/10 with code 3 unused).
3. **Should become parameters:** dot length K (and tile), accumulator width (function of K and activation range), weight-code remap and scale convention at the loader boundary, head precision and vocabulary in system models, context length, activation bit width (as an analysis parameter now, Q-021).
4. **INT8 still a reasonable first precision:** yes, as current cross-model evidence from both production-like families.
5. **Configurable activation width later:** keep it as an open design parameter (Q-021); do not redesign now.
6. **256-wide tiling:** more justified than after EXP-0002 (five K values, GCD 256, 6912 excludes 512) but still MEASURED-divisibility, not proven optimality. Deserves investigation as the default tile.
7. **Accumulator widths parameterized by K:** yes; 19–22 bits across observed K, 16–17 per 256-tile.
8. **Scale handling outside the primitive:** confirmed in both families.
9. **Sparse hardware:** less compelling cross-model. Unstructured ~40% zeros everywhere; the only structure (dead channels) vanished in Falcon.
10. **Mixed-precision system thesis:** strengthened. Both families are ternary only in their blocks; BF16 head, norms, attention math, and KV cache remain, and which one dominates is model- and context-dependent.

### Effect on the planned roofline experiment

Model the envelope, not one point:

| Parameter | Observed range | Treat as |
|---|---|---|
| Dot length K | 2048–13312 (two per model: hidden, intermediate) | sweep; tile 256 default, 128/512 variants |
| Output N per projection | 256–13312 | from model specs |
| Layers | 24–32 | parameter |
| Intermediate/hidden | 2.7–6.5 | derived |
| Activation bits | 8 now; 4 as risk case | parameter |
| Zero rate | 38–47% per module; 39.7–42.2% global | parameter; unstructured |
| Head | BF16, 67–328 M params, vocab 32k–128k, tied or untied | parameter incl. INT8/INT4 hypotheticals |
| Context | 4k–32k | sweep; KV bytes/token/layer 1.0–2.6 KB |
| Batch and phase | batch 1 decode (the stream-once assumption) | default batch 1; sweep batch size and prefill vs decode, since batching amortizes the weight stream and prefill is compute-bound |
| GQA | 4:1–8:1 | from spec |
| Packing | 2-bit as stored; 1.6-bit hypothetical | parameter |
| Scale convention | β or 1/β at load; no runtime effect | ignore in roofline |

Inputs can be read directly from the three `model_summary.json` files.

## Surprises

- Falcon-E has **no** dead channels at all. BitNet's 4.76% is a recipe artifact.
- Falcon stores 1/β; the same transformers package serves two opposite checkpoint conventions, selected by one config field.
- BF16 emulation reproduces Falcon's packed codes exactly; the same emulation makes BitNet's agreement worse.
- At 32k context, Falcon's KV read (805–1,074 MB/token) exceeds its entire weight stream.
- K = 2048 saves an accumulator bit only because -128 is unreachable.

## Limitations

- Two families, five K values, three checkpoints. Nothing here is universal.
- Both inference activation quantizers are the same transformers code; independence rests on TII's training kernel.
- All traffic and MAC figures are static; nothing was run.
- Falcon 3B was not cross-checked against latents (sharded branch); its scale convention is inferred from the shared code path and the stored-scale magnitudes (5–58, consistent with 1/β).
- Falcon's license permits analysis; this experiment produces statistics, not derivative weights.

## What This Proves

- The listed invariants hold on two independent native ternary families at the stated revisions.
- The generalized profiler reproduces EXP-0002 exactly.
- Falcon-E's packed checkpoint is exactly the official ternarization of its latent branch.

## What This Does NOT Prove

- That any property holds for a third family or a future model.
- Optimality of 256-wide tiles.
- Any hardware performance, area, power, or bandwidth figure.
- That INT8 activations will remain the production norm.

## Artifacts / Logs

- `results/workload_profile/cross_model/comparison.md`, `comparison.json`
- `results/workload_profile/falcon-e-1b/` and `falcon-e-3b/` (15 files each; 1B includes `ternarization_crosscheck.json`)
- `results/workload_profile/` (BitNet; unchanged numbers, extended columns; `ternarization_crosscheck.json` now also reports BF16-emulated agreement)
- `analysis/model_specs.py`, `ternary_model_profile.py`, `ternarization_crosscheck.py`, `cross_model_compare.py`; `tests/test_analysis_cross_model.py`; `scripts/run_workload_profile.ps1`

## Follow-up Questions

- Q-018 (third family), Q-019 (head and KV scope), Q-021 (activation precision parameterization).
- Is the rising-with-depth sparsity profile in Falcon a SiLU effect, and does it matter for anything hardware-visible? (No action unless a use appears.)
