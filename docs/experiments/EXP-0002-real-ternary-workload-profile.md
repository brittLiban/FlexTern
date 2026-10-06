# EXP-0002 — Real ternary transformer workload profile (BitNet b1.58 2B4T)

Status: Complete
Date: 2026-10-06
Commit: the Phase 2 commit ("research: characterize real ternary transformer workload"); results generated from that tree.

All tables below are copied from the generated `results/workload_profile/workload_summary.md`. That file and the CSV/JSON beside it are the primary record; this document interprets them. Labels: KNOWN (external fact / reference code), MEASURED (counted from the checkpoint), INFERRED (computed from measured dimensions and format), HYPOTHESIZED, SPECULATIVE.

## Research Question

What does a real, publicly released, natively ternary transformer require from the hardware, and which architectural bottleneck should FlexTern attack next?

## Hypothesis

Not a pass/fail hypothesis. The working expectations going in: dot lengths in the low thousands; zero rates well above 0; INT8-like activations; some scale handling around the ternary matmul; memory traffic plausibly dominant at batch 1.

## Why It Matters

Phase 1 proved the primitive is cheap and that accumulation dilutes it. Every next hardware choice (accumulator, sparsity, packing, activation width, dataflow, memory) depends on what the real workload looks like. Guessing would create architectural debt.

## Setup

**Model (ADR-0003):** `microsoft/bitnet-b1.58-2B-4T`, Hugging Face revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`, file `model.safetensors`, 1,178,623,988 bytes, SHA-256 `8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b` (matches the Hugging Face linked-etag). License MIT. Ungated. Paper: BitNet b1.58 2B4T Technical Report, arXiv:2504.12285. Model card states: native 1.58-bit weights via absmean quantization, trained from scratch (not post-training quantized), 8-bit per-token absmax activations ("W1.58A8"), 4 T training tokens, 4096 context, LLaMA-3 tokenizer, subln normalization, ReLU² FFN, no bias terms.

**Weight format on disk (MEASURED from header):** 542 tensors: 210 `U8` packed ternary weights, 210 `BF16` scalar `weight_scale`, 121 `BF16` norm vectors, 1 `BF16` embedding `[128256, 2560]`. No `lm_head.weight` tensor exists; `config.tie_word_embeddings = true`. Weights are stored literally as ternary codes: 2 bits each, code = q + 1, four per byte packed along the **output** dimension (packed shape `[N/4, K]`). Code 3 never occurs (0 of 2.08 B values).

**Reference quantization code (KNOWN):** `huggingface/transformers` `src/transformers/integrations/bitnet.py` @ `d6abd1333078195535f461589b64f112b53688b2`. `config.quantization_config.linear_class = autobitlinear`, `quantization_mode = offline`, so `AutoBitLinear.forward` applies: `output = F.linear(x, W_q) * weight_scale`. Activation quantization (`ActQuant` / `activation_quant`): `scale = 127 / absmax(x, last dim)`, `round`, `clamp(-128, 127)`, dequantize by dividing by the same scale. Weight ternarization during training (`WeightQuant`): `scale = 1 / mean|W|`, `round(W * scale).clamp(-1, 1)`.

## Hardware / Software Environment

Windows 11 Home 10.0.26200; Python 3.12.10; numpy 2.5.3; huggingface_hub 2.1.1. No GPU, no torch. Analysis is numpy over a memory-mapped file; the 657 MB embedding bytes are never read. Runtime ≈ 40 s from cache. Deterministic; no seeds. Full record: `results/workload_profile/environment.txt`.

## Baseline

Not a comparison experiment. Where useful, BF16/INT8/INT4 storage equivalents are computed as THEORETICAL references.

## Method

1. Hugging Face API + header-only HTTP range read to verify provenance, size, dtypes, and shapes before downloading (`analysis/`, this record).
2. Pinned-revision download to the Hugging Face cache (outside the repo; `.gitignore` excludes checkpoints).
3. `analysis/ternary_model_profile.py`: parse header, memmap, `verify_layout()` proves `8 + header + Σ tensor bytes == file size` with no gaps or overlaps; classify every tensor into exactly one of {packed ternary, weight_scale, norm, embedding} and assert the unclassified list is empty and counts are 210/210/121/1; assert packed rows × 4 and columns equal the config-derived N and K for every projection; unpack each ternary tensor with the transcribed `unpack_weights` logic; count -1/0/+1/code-3 and assert they sum to the element count; assert the global total equals the config-derived 2,084,044,800; per-row zero statistics; aligned-group all-zero rates along K for g = 4, 8, 16 versus p^g; all-zero rows/columns; MLP dead-channel intersection of gate/up rows with down columns; dot-length distribution; static operation inventory per block; storage and traffic tables; accumulator widths.
4. `analysis/ternarization_crosscheck.py`: HTTP-range fetch of three BF16 latent tensors from `microsoft/bitnet-b1.58-2B-4T-bf16` (revision `27668139…`), apply `WeightQuant` semantics, compare with the unpacked packed-repo codes and the stored scale.
5. `tests/test_analysis.py` (16 tests): the transformers docstring unpack example verbatim; pack/unpack round trip; exact counting; code-3 surfaces as +2 rather than being folded; BF16 decode on known bit patterns; synthetic safetensors file shape/byte/layout checks and truncation detection; `signed_bits` and accumulator-width formula against brute-force enumeration at K = 1, 2 (full INT8 range) and K = 3 (reduced range); the Phase 1 reference (K = 16 → 13 bits); reproduction of `activation_quant` showing -128 unreachable; reproduction of absmean ternarization on a known matrix.
6. `scripts/run_workload_profile.ps1` runs tests → profile → cross-check and stops on failure.

## Metrics

Parameter counts by class; -1/0/+1 counts and rates (global, per tensor, per module, per layer); zero structure along K; dead rows/columns; dot-length histogram weighted by weights; static MAC/op counts per block; storage bytes by representation; bytes per token under a stream-once decode model; signed accumulator bits by K.

## Results

### Architecture (MEASURED from config.json)

30 layers; hidden 2560; intermediate 6912; 20 heads × 128 = 2560; 5 KV heads (GQA 4:1), so k/v project to 640; vocab 128,256; context 4096; RoPE θ = 500,000; RMSNorm ε = 1e-5 with four norms per layer (input, post-attention, `attn_sub_norm` before o_proj, `ffn_sub_norm` before down_proj) plus a final norm; activation ReLU²; MLP `down(ffn_sub_norm(relu2(gate(x)) * up(x)))`; tied embeddings; no bias anywhere.

### Parameters (MEASURED)

| Class | Count | % |
|---|---:|---:|
| Ternary linear weights | 2,084,044,800 | 86.4 |
| ├ attention q,k,v,o | 491,520,000 | 20.4 |
| └ MLP gate,up,down | 1,592,524,800 | 66.0 |
| Embedding BF16 (tied lm_head) | 328,335,360 | 13.6 |
| Norms BF16 | 440,320 | 0.018 |
| Weight scales BF16 | 210 | ~0 |
| Total | 2,412,820,690 | 100 |

### Ternary distribution (MEASURED)

| Value | Count | % |
|---|---:|---:|
| -1 | 602,163,685 | 28.894 |
| 0 | 879,693,294 | 42.211 |
| +1 | 602,187,821 | 28.895 |

+1 and -1 differ by 24,136 values (0.004%): balanced to four decimal places. Per-tensor zero rate: min 36.11% (`layers.4.self_attn.v_proj`), median 41.16%, max 60.08% (`layers.1.mlp.gate_proj`), std 4.23 points. By module: q 46.75%, k 44.65%, v 37.64%, o 41.84%, gate 42.36%, up 41.61%, down 41.31%. By layer: layer 1 is 56.2% zero (MLP 60.0%), layer 2 49.1%, layer 3 44.6%; layers 4–29 sit at 40.7–42.8%. Sparsity is **mildly module-dependent** (v least sparse, q most) and **strongly layer-dependent only in the first three layers**.

### Zero structure (MEASURED)

Aligned g-groups along K that are all zero, size-weighted, versus p^g under independence:

| Module | g=16 measured | g=16 p^16 | g=16 excluding all-zero rows |
|---|---:|---:|---:|
| gate_proj | 0.047599 | 0.000011 | 0.000001 |
| up_proj | 0.047599 | 0.000011 | 0.000000 |
| down_proj | 0.000011 | 0.000011 | 0.000011 |
| q_proj | 0.004521 | 0.000007 | 0.004469 |
| o_proj | 0.004043 | 0.000003 | 0.004043 |

In the MLP, every excess over independence is explained by entirely zero rows. In q/k/o the small excess remains after removing dead rows and comes from rows with very high individual zero rates (row zero-rate std 0.13 for q), not from clustering along K.

**Dead channels:** 9,870 of 207,360 MLP intermediate channels (4.76%) have both their gate_proj and up_proj rows entirely zero; the sets coincide exactly in every layer, and the matching down_proj columns are zero too (10,018 dead columns). Concentration: layer 0 4.77%, layer 1 43.59%, layer 2 27.58%, layer 3 14.77%, layer 4 8.91%, layers 5–29 between 0.3% and 4.4%. v_proj has 103 dead rows (79 in layer 29); q_proj 4; k and o none. o_proj has 1,176 all-zero input columns: under 4:1 GQA the 103 dead v rows account for 412 of them, and the remaining 764 are attention-output dimensions the model learned to ignore. 2.44% of all ternary weights sit in dead rows.

### Scale (KNOWN from code, MEASURED from data)

One BF16 scalar per linear, range 0.746–4.594 across 210 tensors. Cross-check against the BF16 latent repo on three tensors: stored `weight_scale` equals `mean|W_latent|` to BF16 precision (1.21875 vs 1.21885; 1.79688 vs 1.79406; 2.68750 vs 2.68310), confirming `weight_scale = β` and `W_eff = β · W_q`. The latent weights simply have mean magnitude 1–3. Applying `WeightQuant` to the latents reproduces the packed codes at 98.78%, 99.37%, and 98.01% respectively (`results/workload_profile/ternarization_crosscheck.json`). The method is confirmed; the two repos are not bit-identical snapshots (KI-004). The packed repo is the deployed artifact and is what this experiment measures.

Scale is applied once per output element after accumulation (`* weight_scale`, then `/ activation_scale`). It never enters the dot product. Per layer: 22,784 scale multiplies against 69,468,160 ternary MACs (0.03%).

### Activations (KNOWN from code)

Per-token symmetric INT8, dynamic absmax per linear input, `scale = 127/absmax`. Since |x·scale| ≤ 127 before rounding, **-128 is unreachable**; the effective range is [-127, 127] (reproduced in `tests/test_analysis.py`). Four distinct quantizations per layer (q/k/v share an input, o, gate/up share, down). Each requires a full pass over the vector (absmax) before any element can be quantized.

### Dot lengths (MEASURED)

| K | Modules | Dot products per token | Weights | % | % zero | E[nonzero terms] |
|---:|---|---:|---:|---:|---:|---:|
| 2560 | q,k,v,o,gate,up | 606,720 | 1,553,203,200 | 74.5 | 42.52 | 1,472 |
| 6912 | down | 76,800 | 530,841,600 | 25.5 | 41.31 | 4,057 |

Exactly two dot lengths exist. Both are multiples of 256 (2560 = 10 × 256; 6912 = 27 × 256) and of 128. Representative matrix shapes per layer: 2560×2560 (q, o), 640×2560 (k, v), 6912×2560 (gate, up), 2560×6912 (down). A dot of length 2560 has on average ~736 +1 terms, ~736 -1 terms, ~1,088 zeros.

### Operation mix per token (INFERRED, static)

Ternary linear MACs 2,084,044,800 (86.4% of linear MACs). Tied BF16 output head 328,335,360 MACs (13.6%), K = 2560, N = 128,256, non-ternary. Attention scores at 4096 context: 629,145,600 BF16 MACs (QKᵀ + AV), growing linearly with context. Per layer, non-linear work is small in op count: activation quant 29,184 element-ops, RMSNorm 14,592 elements, RoPE 6,400 elements, softmax 81,920 scores at 4096, ReLU² and gate×up 6,912 each, residual adds 5,120, scale multiplies 22,784. KV cache: 2,560 bytes written per token per layer; 10.5 MB read per token per layer at 4096 context.

### Storage (MEASURED file / THEORETICAL)

| Representation | Ternary-layer bytes |
|---|---:|
| BF16 | 4,168.1 MB |
| INT8 | 2,084.0 MB |
| INT4 | 1,042.0 MB |
| 2-bit (as stored) | 521.0 MB |
| 5 trits/byte | 416.8 MB |
| log2(3) bound | 412.9 MB |

Actual file 1,178.6 MB: 521.0 MB packed ternary (44.2%), 656.7 MB BF16 embedding/head (55.7%), 0.9 MB norms, 420 bytes of scales. A 2-bit-layers + INT8-head model would be 850.2 MB; with an INT4 head, 686.1 MB (HYPOTHETICAL; quality unknown).

### Decode traffic (INFERRED / ANALYTICAL, batch 1, stream every weight once per token)

| Scenario | Weight bytes/token | % head | + KV read @ 4096 |
|---|---:|---:|---:|
| As stored: 2-bit layers + BF16 head | 1,178.6 MB | 56 | 1,493.1 MB |
| BF16 dense reference | 4,825.6 MB | 14 | 5,140.2 MB |
| INT8 layers + BF16 head | 2,741.6 MB | 24 | 3,056.2 MB |
| 2-bit layers + INT8 head | 850.2 MB | 39 | 1,164.8 MB |
| 2-bit layers + INT4 head | 686.1 MB | 24 | 1,000.6 MB |
| 2-bit layers only (head excluded) | 521.9 MB | 0 | 836.5 MB |

Invalidated by on-chip residency, batching, weight reuse, compression, sharding, speculative decoding, or a quantized head.

### Accumulator widths (INFERRED mathematically; formula validated by brute force at K = 1, 2, 3)

| K | Format [-128,127] sum range | bits | Model [-127,127] sum range | bits | Max nonzero/row in data | data-bounded bits |
|---:|---|---:|---|---:|---:|---:|
| 2560 | ±327,680 | 20 | ±325,120 | 20 | 2560 | 20 |
| 6912 | ±884,736 | 21 | ±877,824 | 21 | 6785 | 21 |
| 256 (tile) | ±32,768 | 17 | ±32,512 | 16 | | |
| 16 (Phase 1) | ±2,048 | 13 | ±2,032 | 12 | | |

The format/model distinction changes the width only at tile scale (16, 64, 256, 512, 1024 all save one bit under the model range) and never at full K. At least one K = 2560 row is fully dense (2560 nonzeros), so data-bounded widths equal format widths at full K.

## Interpretation

### Y-Core v0.1 assumptions versus the real model

1. **INT8 activation representative?** Yes (KNOWN). Signed INT8, per-token absmax. The model never emits -128, so Y-Core's +128 output case is unreachable on this workload but the format still requires 9 bits.
2. **2-bit ternary encoding representative?** Yes (MEASURED). Weights are stored literally as 2-bit codes with the same three-valued semantics. The checkpoint's code assignment (00/01/10 = -1/0/+1 after the +1 offset) differs from ADR-0002's two's-complement style (00/01/11); conversion is a trivial remap, not a structural difference.
3. **Scaling required around Y-Core?** Yes but outside it (KNOWN): one per-tensor multiply and one per-token divide per output element, after accumulation.
4. **How frequently?** Once per output element: 22,784 per layer versus 69.5 M MACs (0.03%). Plus four absmax passes per layer for activation quantization.
5. **Does scale handling change the primitive?** No. The dot product is pure integer. The pipeline dependency of dynamic absmax (whole vector before quantizing) is a dataflow concern, not a datapath one.
6. **Zero rates high enough for immediate zero-skipping investigation?** 42% zeros is high, but placement along K is independent (no g = 4/8/16 blocks beyond chance once dead rows are removed). Per-element skipping would have to exploit unstructured zeros at a cost that EXP-0002 cannot evaluate. Channel-level dead sparsity (4.76%) is real, exact, and skippable by a compiler with no hardware. Verdict: not immediate; evidence favors handling zeros cheaply inside accumulation (a zero term is a no-op) over sparse dataflow.
7. **Required accumulation widths:** 20 bits (K = 2560), 21 bits (K = 6912) under the INT8 format; the same under the model range. A 256-wide tile needs 17 (format) or 16 (model) bits.
8. **Separate ±accumulation compatible?** Yes and naturally balanced: +1 and -1 occur at 28.89% each, so two accumulators would see equal loads. Each would need one bit less than the signed total at the tile level (sum of up to 2560 × 127 = 325,120 → 19 bits unsigned).
9. **Bit-plane/popcount worth investigating?** Plausibly (INFERRED): activations are exactly 8 bits, weights are sign/zero masks, K is a multiple of 256, and ± are balanced; a dot product is Σ_b 2^b (popcount(pos ∧ x_b) − popcount(neg ∧ x_b)) over 8 planes with sign handling for the top bit. Whether that beats a chain/tree of 9-bit terms is unmeasured.
10. **Memory before arithmetic?** Under the stream-once model, yes: 1,179 MB of weights plus 315 MB of KV per token at 4k context versus 2.08 GMAC of ternary work. But 56% of weight bytes are the non-ternary head. The ternary stream is already at 2 bits; FlexTern can shave 20% with 5-trit packing, not more. Whether memory dominates on real hardware depends on bandwidth and residency, which are unmeasured (SPECULATIVE if quantified).

### Candidate next experiments, ranked by evidence from this model

| Rank | Candidate | Why it might matter | Evidence | Remaining uncertainty |
|---|---|---|---|---|
| 1 | **B/C/A combined: accumulation architecture at real K** (chain vs balanced tree vs carry-save vs dual-sign ±accumulators, 256-wide tiles composing to 2560/6912, 20–21-bit outputs) | EXP-0001 showed accumulation is ~60% of Y-Dot16 logic at K = 16; real K is 160–430× longer with 20–21-bit sums, so accumulation share can only grow. ± balance makes dual accumulators natural and would remove the per-element negate. | MEASURED K, widths, balance; MEASURED EXP-0001 cell split | Which structure wins, and by how much; logic depth not yet measured |
| 2 | **H: memory-feed / weight-stream architecture** | Bytes, not MACs, plausibly dominate batch-1 decode. Packing is 4 output rows per byte at one K position, which dictates that a byte feeds 4 dot products in parallel. | INFERRED traffic model; MEASURED packing layout | Bandwidth is hardware-specific; 56% of the stream is the head, whose scope is a founder decision (Q-019) |
| 3 | **F: bit-plane / popcount execution** | Exact INT8 activations, mask-like weights, K multiple of 256, balanced ±. Could replace 9-bit adders with popcounts. | INFERRED from format | Entirely unmeasured; may be worse at 8 bits than at 4 |
| 4 | **E: ternary packing (5 trits/byte)** | 20% fewer ternary bytes (521 → 417 MB) | THEORETICAL | Base-3 decoder cost vs saving; only 44% of stream is ternary |
| 5 | **J: channel-dead exploitation** (compiler/packing, not hardware) | 4.76% of MLP channels, 43.6% in layer 1, exactly skippable | MEASURED | Small globally; a scheduling task, not RTL (Q-020) |
| 6 | **D: per-element zero skipping / sparse dataflow** | 42% zeros | MEASURED, but unstructured | Evidence is against hardware block-skipping; cost of detecting unstructured zeros unknown |
| 7 | **I: scale handling** | Must exist | KNOWN | Negligible: 0.03% of ops, outside the dot product |
| 8 | **G: activation precision change** | Would shrink activation traffic and accumulators | None from hardware | Model-fixed at INT8; quality study needed first; not a FlexTern-only question |

### Recommended single next objective

**EXP-0003: accumulation architecture at real dot lengths.** Build parameterized Y-Dot variants (sequential chain as in v0.1, balanced tree, carry-save, dual-sign ±accumulators with one final subtract) at a 256-wide tile with 17-bit partial sums, composed to K = 2560 and 6912 with 20/21-bit outputs; synthesize generically; report cells and logic depth; keep the INT8 dot16-style baseline scaled to the same K for reference. Include bit-plane popcount only if it can be expressed in the same harness without widening scope.

Why this and not the memory stream: the memory finding is real but its largest component is outside the ternary datapath and needs a scope decision (Q-019) before any experiment is well-posed. Accumulation is entirely within FlexTern's control, is the dominant logic in Y-Dot16 already, and every later Y-Array decision depends on it.

## Surprises

- The non-ternary output head is 13.6% of parameters but 55.7% of bytes on disk and the single largest per-token weight stream. The "1.58-bit model" is 1.58-bit only in its transformer blocks.
- Dead MLP channels concentrate in the first four layers (43.6% in layer 1), with gate and up dead sets identical in every layer.
- Latent weights have mean magnitude 1–3, so the stored scales are 0.7–4.6 rather than hundredths.
- -128 cannot occur in the model's activations.
- At least one weight row is fully dense (0 zeros in 2560), so data-bounded accumulator widths equal format widths.

## Limitations

- One model, one revision. Generalization is Q-018.
- All MAC, op, and byte-per-token figures are static. No runtime was measured; no cache behavior, bandwidth, or latency is known.
- The traffic model's stream-once assumption is the most pessimistic one for weights and ignores batching entirely.
- Ternarization cross-check reached 98–99.4% agreement, not 100%; the BF16 repo is a different snapshot. Nothing in this experiment depends on the BF16 repo.
- Zero-structure analysis covers aligned groups of 4/8/16 along K and whole rows/columns only. Other structures (per-head, per-block-of-rows) were not examined because the simple statistics matched independence.
- Attention arithmetic is reported at 4096 context; it is proportional to context and dominates linear MACs only at long contexts.

## What This Proves

- The reference workload's shapes, dot lengths, distributions, formats, and scale mechanics, as listed above, for this exact revision.
- That Y-Core v0.1's input/output format matches the model's datapath format.
- That zero placement along K is statistically independent in this model once dead rows are removed.
- That the analysis is reproducible from a clean checkout with one command and verifies its own input.

## What This Does NOT Prove

- Any hardware performance, area, power, or bandwidth figure.
- That memory dominates on any real device.
- That any candidate experiment will yield a saving; the ranking is a judgment on evidence, not a measurement.
- That findings transfer to other ternary models.

## Artifacts / Logs

- `results/workload_profile/workload_summary.md` (generated), `model_summary.json`, `weight_distribution.csv` (210 rows), `weight_distribution_by_module.csv`, `weight_distribution_by_layer.csv`, `dead_channels_by_layer.csv`, `layer_shapes.csv`, `dot_length_summary.csv`, `expected_terms_per_dot.csv`, `operation_inventory.csv`, `memory_representation.csv`, `decode_traffic_model.csv`, `accumulator_widths.csv`, `ternarization_crosscheck.json`, `environment.txt`.
- Code: `analysis/ternary_model_profile.py`, `analysis/safetensors_reader.py`, `analysis/ternarization_crosscheck.py`, `tests/test_analysis.py`, `scripts/run_workload_profile.ps1`.
- Model checkpoint: Hugging Face cache only; not committed.

## Follow-up Questions

- Q-018 (second model), Q-019 (head scope), Q-020 (channel-dead exploitation at compile time), Q-009/Q-010 (accumulation structure, now with real K and widths).
- Does the model's dynamic per-token absmax create a pipeline bubble that matters more than the quantization arithmetic itself?
- Is the packing-along-N layout (one byte = four output rows) the right memory layout for a Y-Array, or should FlexTern repack along K?
