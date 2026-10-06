# EXP-0004 — Low-precision output head and KV cache on a native ternary model (Falcon-E 1B)

Status: Complete (awaiting founder/ChatGPT review)
Date: 2026-10-06
Commit: 438da28 ("research: evaluate INT4 head and KV cache on Falcon-E 1B (EXP-0004)"); results generated from that tree.

Primary data: `results/model_precision/` (JSON per phase, `quality_comparison.csv`, `memory_comparison.csv`, `latency_comparison.csv`, `candidate_format.md`, `roofline_reconnect.md`). This record interprets them. Labels as in `AGENTS.md` §7. **This is a software experiment on one checkpoint; nothing here is a hardware result.**

## Research Question

Can `tiiuae/Falcon-E-1B-Base` (ADR-0004 revision) preserve acceptable language-model quality when (A) its BF16 output head is stored as INT4, (B) its KV cache is stored at ~4 bits, and (C) both, and what do the real representations (packing, metadata, dequantization, cache semantics) cost?

## Hypothesis

H6 (THESIS, HYPOTHESIZED): co-designing low-precision non-ternary components with the ternary blocks improves end-to-end economics. This experiment tests only the model-side premise: that the non-ternary memory footprint can be cut substantially without losing useful behaviour. No acceptable-loss threshold is predeclared; the tradeoff frontier is reported.

## Why It Matters

EXP-0003: for Falcon-E 1B per generated token, ternary weights are ~396 MB, the BF16 head 134 MB, and BF16 KV reads up to 805 MB at 32k context under stream-once accounting. The ternary engine cannot touch the head or the KV cache. If they can be stored at ~4 bits with acceptable quality, the system ceiling changes; if not, they remain hard limits.

## Setup

**Model (ADR-0004, MEASURED at load):** `tiiuae/Falcon-E-1B-Base` revision `f4001b8b…`, `model.safetensors` 665,041,488 bytes, SHA-256 `f62c270b…c79d` verified before every run. Loaded through transformers' BitNet integration: all 168 ternary projections are `BitLinear` modules holding packed uint8 weights (q_proj `[512, 2048]` uint8, `weight_scale` BF16 = 17.375 for layer 0), 396,361,728 bytes resident; `lm_head` is a plain `nn.Linear` BF16 `[32768, 2048]` (134,217,728 bytes), **untied** from `embed_tokens`; attention implementation `sdpa`; 665 MB allocated on the GPU after load.

**Reference KV cache (MEASURED, not assumed):** `DynamicCache` of `DynamicLayer`s, keys and values BF16 of shape `[1, 2, T, 128]` per layer (2 KV heads × 128 head dim, GQA 8:1). This confirms EXP-0003's BF16-KV assumption from data.

**Configurations** (ternary transformer weights unchanged in all):

| Config | Head | KV cache |
|---|---|---|
| baseline | BF16 `nn.Linear` | BF16 `DynamicCache` |
| A | INT4 RTN, g = 128 | BF16 `DynamicCache` |
| B | BF16 | 4-bit `QuantizedCache` (hqq, g = 64, transformers default axis 0, 128-token BF16 residual) |
| B1 | BF16 | 4-bit `QuantizedCache` (hqq, g = 64, axis 1 = per-token groups along head_dim, 128-token residual) |
| C | INT4 RTN g128 | 4-bit `QuantizedCache` axis 1 |

**Head quantizer (`analysis/model_precision/head_quantization.py`):** weight-only, round-to-nearest, asymmetric, groupwise along the input dimension (K = 2048, 16 groups of 128 per output row): fp16 scale = (max − min)/15, zero = round(−min/scale) ∈ [0, 15], codes packed two per byte (low nibble = even column). Execution dequantizes to BF16 every forward and calls `F.linear`. Original head kept on CPU for baseline runs.

**KV quantizer:** transformers v5.18.0 `QuantizedCache` (KIVI-style): prefill quantized at lazy initialization; new tokens accumulate in a BF16 residual; when the residual reaches 128 tokens the whole history (dequantized old + residual + new) is re-quantized; every `update` dequantizes the entire quantized history into a BF16 temporary and concatenates it for attention. Backend `hqq` (pure PyTorch; the quanto backend needs a CUDA toolkit to JIT its extension, KI-006). 4-bit asymmetric, group 64, BF16 scale and zero per group, packed `4bit_u8`.

## Hardware / Software Environment

WSL2 Ubuntu (kernel 6.18.33-microsoft-standard) on Windows 11, because Smart App Control blocks PyTorch's unsigned DLLs on the host (KI-005). NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, compute capability 12.0, driver 596.08 via WSL passthrough. Python 3.14.4; torch 2.11.0+cu128; transformers 5.18.0; hqq 0.2.8.post1; optimum-quanto 0.2.7 (present, unusable here); lm_eval 0.4.13; datasets 5.1.0; accelerate 1.15.0; numpy 2.5.3. `TORCHDYNAMO_DISABLE=1` (no C++ compiler for `torch.compile`, so `BitLinear` runs eagerly and unpacks its weights every forward). WSL memory capped at 6 GB; host RAM 15.3 GB shared with other applications. Seed 20261006 for all harness runs; greedy decoding only.

## Baseline

The unmodified model (BF16 head, BF16 cache). For lm-eval tasks, the baseline run in this exact harness and settings is the reference; the model card publishes only leaderboard-v2 tasks (IFEval, BBH, MATH-Hard, GPQA, MuSR, MMLU-Pro), so there is no external anchor for these tasks.

## Method

1. Feasibility audit (`audit.py`): identity, load path, module classes and dtypes, default cache dtype and shape, greedy sample, quantized-cache internals after a 382-token generation.
2. Unit tests (`tests/test_model_precision.py`): pack/unpack on known codes, error ≤ scale/2, exact reconstruction of representable weights, matmul vs dequantized reference, logical-bytes accounting; synthetic `HQQQuantizedLayer` checks: first-update passthrough, append, GQA shapes, re-quantization drift for axis 0 and axis 1.
3. lm-eval-harness, 0-shot, batch 4, `max_length` 2048 (rolling-perplexity window; the model's 32k maximum would require 32k × 32k logits): `wikitext` (word perplexity), `hellaswag`, `piqa`, `arc_easy`, `arc_challenge`, `winogrande`. Run for baseline and A on all tasks; B on `wikitext` only to demonstrate that loglikelihood scoring (one forward pass, no incremental cache) is invariant to the KV setting. A 20-sample pilot validated the pipeline first.
4. Teacher-forced incremental-decode perplexity through the real cache (`evaluate.incremental_nll`): fixed WikiText-2 test windows (first N non-overlapping windows of the tokenized test split), prefill 128 tokens, then one token per step with the chosen cache; NLL accumulated for positions ≥ 128 and bucketed by position. Control: the baseline incremental run must reproduce the single-forward NLL on the same windows.
5. Greedy generation samples at short, medium (1,500-token) and long (6,144-token) prompts; KV memory sweep over 128–16,384 context with cache bytes from the stored tensors and `max_memory_allocated`; decode latency with 3 warmups and 5 repeats.
6. Report generation and EXP-0003 scenario re-evaluation with the measured bits per element (`roofline_reconnect.py`; `results/roofline/` untouched).

## Metrics

Task accuracies (acc, acc_norm) with stderr and sample counts; word perplexity; incremental perplexity with per-position buckets; logical bytes, resident bytes, dequantization temporaries, peak allocation; prefill and decode latency.

## Results

All MEASURED unless labeled. PILOT labels mark subsets. Values quoted from `results/model_precision/`; relative changes are versus the baseline row of the same table. One checkpoint, one run per cell (lm-eval loglikelihood scoring and greedy teacher-forced decode are deterministic).

### R1. Head quantization (INT4 RTN g128) — `head_int4.json`

| Quantity | Value |
|---|---|
| Original BF16 head | 134,217,728 B |
| Logical INT4 bytes (codes + fp16 scale + 4-bit zero) | 34,865,152 B = 4.156 bits/element; metadata 1,310,720 B |
| Resident as stored here (zero kept in uint8) | 35,127,296 B |
| Dequantization temporary per forward | 134,217,728 B BF16 weight (plus float32 math temporaries; observed decode-step peak +0.5 GB vs baseline) |
| Weight reconstruction error | relative Frobenius 0.1057; max abs 0.03125; max error / (scale/2) = 1.073 (fp16 scale rounding pushes a few elements marginally past half a step) |

### R2. lm-eval-harness (0-shot, batch 4, `max_length` 2048) — `lm_eval.json`, `quality_comparison.csv`

Full test/validation sets except HellaSwag (PILOT: first 2,000 of 10,042). Stderr from the harness.

| Task (n) | Metric | baseline | A (INT4 head) | Δ |
|---|---|---:|---:|---:|
| wikitext (62 docs, full test) | word perplexity | 21.254 | 21.396 | +0.67% |
| wikitext | bits per byte | 0.8246 | 0.8264 | +0.0018 |
| piqa (1,838) | acc | 0.6997 ± 0.0107 | 0.6997 | 0.00 pp |
| piqa | acc_norm | 0.6980 | 0.6964 | −0.16 pp |
| arc_easy (2,376) | acc | 0.6726 ± 0.0096 | 0.6684 | −0.42 pp |
| arc_easy | acc_norm | 0.6149 | 0.6136 | −0.13 pp |
| arc_challenge (1,172) | acc | 0.3712 ± 0.0141 | 0.3720 | +0.08 pp |
| arc_challenge | acc_norm | 0.3882 | 0.3925 | +0.43 pp |
| winogrande (1,267) | acc | 0.5714 ± 0.0139 | 0.5801 | +0.87 pp |
| hellaswag (2,000, PILOT) | acc | 0.3790 ± 0.0109 | 0.3785 | −0.05 pp |
| hellaswag (PILOT) | acc_norm | 0.4890 ± 0.0112 | 0.4930 | +0.40 pp |

Every task difference is inside one standard error. **Invariance control:** config B (4-bit KV, BF16 head) on wikitext reproduced the baseline exactly (21.2535 / 1.7711 / 0.8246), confirming that loglikelihood scoring never touches the KV cache; B1 and C were therefore not re-run on lm-eval, and no lm-eval number is KV evidence.

### R3. Teacher-forced incremental decode through the real cache — `incremental_ctx2048_w2.json`, `incremental_ctx4096_w1.json`

Fixed WikiText-2 test windows (`Salesforce/wikitext`, `wikitext-2-raw-v1`), prefill 128, one token per step, token-level NLL over positions ≥ 128 (a different quantity from R2's lm-eval word perplexity). PILOT scale: two 2,048-token windows at tokens 0 and 2048 (3,840 scored tokens) and one 4,096-token window at token 0 (3,968 scored tokens). The 4k window is the same text as the two 2k windows, so 2k → 4k tests context length on identical text, not more data; total unique text scored is 1.1% of the test split.

| Context | Config | Head | Cache | token ppl | Δ vs baseline |
|---:|---|---|---|---:|---:|
| 2048 | single-forward reference | BF16 | none | 7.832 | — |
| 2048 | baseline | BF16 | BF16 dynamic | 7.822 | (−0.13% vs single forward: control) |
| 2048 | A | INT4 | BF16 dynamic | 7.860 | +0.49% |
| 2048 | B | BF16 | int4 hqq g64 axis 0 (strided) | 7.872 | +0.65% |
| 2048 | B1 | BF16 | int4 hqq g64 axis 1 (per-token) | 7.835 | +0.17% |
| 2048 | C | INT4 | int4 axis 1 | 7.872 | +0.65% |
| 4096 | single-forward reference | BF16 | none | 7.649 | — |
| 4096 | baseline | BF16 | BF16 dynamic | 7.638 | (−0.14%: control) |
| 4096 | A | INT4 | BF16 dynamic | 7.675 | +0.49% |
| 4096 | B | BF16 | int4 axis 0 | 7.720 | +1.07% |
| 4096 | B1 | BF16 | int4 axis 1 | 7.664 | +0.34% |

C at 4096 was not run (GPU time; its 2k result equals A + B1 within rounding). Per-position NLL buckets of 256 tokens (`nll_by_position_bucket`): at 4k, B1 differs from baseline by −0.012 to +0.012 nats in every bucket; B differs by −0.013 to +0.046 nats with the two largest differences at positions 1792 and 3328, not at the end of the window. There is no monotone growth of error with position for any configuration (the re-quantization flush happens every 128 tokens, so about 31 flushes occurred at 4k). The incremental baseline sits 0.13–0.14% below the single-forward reference at both contexts, in the same direction: a small systematic difference between the sdpa decode and full-sequence paths, inside the 0.5% control tolerance.

### R4. Greedy generation samples (64 new tokens) — `generation.json`

- Short prompt (11 tokens): all five configurations coherent; A matches baseline token for token until the last word; B/B1 pick "airplane" where baseline picks "television"; C diverges after about 15 tokens and stays coherent.
- Medium prompt (1,500 WikiText tokens): baseline and A identical; B, B1, C diverge after one sentence and remain on-topic, Wikipedia-style.
- Long prompt (6,144 WikiText tokens): **every** configuration, baseline included, falls into a repeated quote-mark loop. Baseline and B produce about 45 coherent tokens first; B1 about 15; A and C about 12. Single sample each; see Interpretation.

### R5. KV-cache memory sweep (one decode step after prefill) — `memory.json`, `memory_comparison.csv`

Cache bytes are read from the stored tensors (`cache_bytes()`): BF16 residual plus packed hqq data, scale and zero. The quantized rows hold a one-token residual after the single decode step.

| Context | BF16 cache | int4 cache (B/B1/C identical) | Ratio | Effective bits/element |
|---:|---:|---:|---:|---:|
| 128 | 3.17 MB | 0.91 MB | 3.49× | 4.59 |
| 512 | 12.61 MB | 3.56 MB | 3.54× | 4.52 |
| 2,048 | 50.36 MB | 14.18 MB | 3.55× | 4.51 |
| 4,096 | 100.69 MB | 28.34 MB | 3.55× | 4.50 |
| 8,192 | 201.35 MB | 56.65 MB | 3.55× | 4.50 |
| 16,384 | 402.68 MB | 113.27 MB | 3.55× | 4.50 |

32,768 was not attempted (a 32k prefill does not fit the 8 GB card alongside the unpacked weights). Peak `max_memory_allocated` during the 16k prefill: baseline 5.46 GB, A 5.36 GB, B 5.21 GB, C 5.08 GB; during the decode step: baseline 4.26 GB, A 4.75 GB, B 4.03 GB, C 4.46 GB. All peaks include the 3.17 GB of ternary weights unpacked once to BF16 (R7). Resident memory after configuration: baseline 3.836 GB, A 3.745 GB (−91 MB measured; the head shrank by 99 MB, the 8 MB gap is unexplained), B 3.880 GB (+44 MB, INFERRED to be hqq working buffers). B's decode-step peak (4.03 GB at 16k) is below baseline's (4.26 GB) because the BF16 dequantization temporary is materialized one layer at a time and freed, whereas R8's "attainable" row charges the full BF16 cache read per token; both are correct and describe different things (peak residency versus bytes moved).

### R6. Latency (median of 5 after 3 warmups, 64 new tokens, cached-unpack mode) — `latency.json`, `latency_comparison.csv`

| Prompt | Config | Prefill (s) | Decode tok/s |
|---:|---|---:|---:|
| 128 | baseline | 0.072 | 14.3 |
| 128 | A | 0.082 | 14.1 |
| 128 | B | 0.573 | 12.2 |
| 128 | B1 | 0.576 | 12.7 |
| 128 | C | 0.637 | 12.4 |
| 2,048 | baseline | 0.334 | 13.9 |
| 2,048 | A | 0.344 | 14.2 |
| 2,048 | B | 1.335 | 12.4 |
| 2,048 | B1 | 1.308 | 12.4 |
| 2,048 | C | 1.296 | 12.1 |

### R7. Ternary unpack-cache control — `unpack_cache_check.json`

Stock `BitLinear` unpacks its 2-bit weights every forward: 6.9 tok/s. Unpacking once to BF16 (3.17 GB resident instead of 396 MB packed) gives 12.6 tok/s with max |Δlogit| = 0.0. R3–R6 used the cached-unpack mode (`--cache-unpacked`); R2 used stock `BitLinear` (no incremental decode involved). Numerics are identical either way.

### R8. EXP-0003 reconnection (ANALYTICAL) — `roofline_reconnect.md`

Falcon-E 1B, batch 1, 100 GB/s, measured 4.156 bits/element head and 4.50–4.59 bits/element KV substituted into the EXP-0003 token model; `results/roofline/` unmodified.

| Context | EXP-0003 reference | A (INT4 head) | B (int4 KV, logical) | C (both, logical) | C attainable on this runtime |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 3.15× | 5.64× | 3.67× | 9.05× | 5.64× |
| 8,192 | 2.18× | 2.68× | 3.07× | 5.32× | 2.68× |
| 16,384 | 1.74× | 1.91× | 2.60× | 3.67× | 1.91× |
| 32,768 | 1.42× | 1.47× | 2.10× | 2.52× | 1.47× |

These are ideal ternary-only ceilings (byte ratios), not speedups. "Attainable" equals A because this runtime dequantizes the whole KV cache to BF16 before every attention call.

## Interpretation

- **INT4 head (INFERRED from R1–R3):** a plain round-to-nearest groupwise quantizer, with no calibration data, costs about 0.5–0.7% perplexity and no task-accuracy change outside one standard error on this checkpoint, while cutting the head from 134 MB to 35 MB logical (3.85×). The head is the component EXP-0003 identified as gating for 128k-vocabulary models; this result says the gate is cheap to open on at least one 32k-vocabulary model. Whether a 128k-vocabulary tied head (BitNet 2B4T) tolerates the same recipe is untested.
- **4-bit KV cache (INFERRED from R3, R5):** the KIVI-style cache with a 128-token BF16 residual costs 0.2–0.7% perplexity at 2k and 0.3–1.1% at 4k, and the grouping axis matters more than the bit width. transformers' default axis-0 grouping stripes groups across tokens, so each flush re-quantizes old tokens in new groups and the error drifts (unit test: 0.35 relative drift on synthetic data; +1.07% ppl at 4k). Per-token grouping along head_dim is idempotent under re-quantization (drift 7e-7; +0.34%). The effective 4.5 bits/element (4-bit codes plus BF16 scale and zero per 64) is the number a hardware format would have to match.
- **Combined (C):** at 2k the combined penalty (+0.65%) is approximately the sum of A and B1, so the two quantizations do not interact measurably at this scale.
- **Long-prompt generation (R4):** the quote loop is a property of greedy decoding on this WikiText prompt (the model, baseline included, is attracted to the quote token that dominates the prompt). Quantization moves the entry point of the loop earlier, which is a brittleness signal worth tracking, but a single greedy sample is not a measurement of quality loss, and the per-position NLL shows no degradation at the same positions under teacher forcing.
- **Memory versus traffic (R5, R6, R8):** resident KV memory shrinks 3.5×, but on this runtime attention still reads BF16 because the cache is dequantized every step, prefill is 4–8× slower because hqq quantizes the prefill, and decode is 11–15% slower. None of the EXP-0003 "logical" ceilings is attainable with this software; the attainable rows equal the INT4-head rows. The traffic benefit of 4-bit KV exists only on an attention path that consumes 4-bit K/V directly (GPU fused kernels exist for this; FlexTern has none; see Competitive baseline).
- **Candidate format (CANDIDATE, not FlexTern v1):** `results/model_precision/candidate_format.md`. Ternary blocks unchanged (2-bit, per-tensor BF16 scale after accumulation, INT8 per-token activations); head INT4 asymmetric g128 with fp16 scale and 4-bit zero; KV 4-bit asymmetric g64 along head_dim with per-group scale and zero, BF16 residual window. Unresolved: whether an INT4-weight head path or a 4-bit-KV attention path can be built without the dequantization temporaries; the acceptable-loss threshold; transfer to other checkpoints.

### Competitive-baseline discipline

Category **A** (available on any GPU today, benefits everyone): both representations. Weight-only INT4 for linear layers and KIVI-style 4-bit KV caches are published, widely deployed GPU techniques; this experiment reproduced them on a native ternary model and invented nothing. Category **B** (requires hardware specific to this format): none demonstrated. Category **C** (FlexTern-only): none. Any FlexTern system claim must therefore compare against a GPU that already uses an INT4 head and 4-bit KV, not against the BF16 reference; by EXP-0003's byte accounting the remaining advantage of 2-bit ternary storage over such a system is the ternary stream only (1.3–1.8× in bytes, EXP-0003 `system_storage_comparison.csv`). No competitor numbers were measured here and none are claimed.

## Surprises

1. The grouping axis of the KV quantizer mattered more than expected: transformers' default is the drifting one, and the per-token alternative is better on every metric measured.
2. The INT4 head costs nothing measurable on the five tasks; the only visible cost is in perplexity at the third significant figure.
3. The long-prompt greedy collapse happens at baseline too. Had only the quantized configurations been sampled, it would have been misattributed to quantization.
4. Stock `BitLinear` re-unpacks ternary weights every forward (6.9 tok/s): the project's "2-bit storage" exists only at rest in this runtime. A one-time unpack to BF16 doubles speed at 8× the resident memory, the software analogue of the EXP-0003 observation that ternary storage only pays if the consumer reads packed bytes.
5. The quanto backend the plan assumed needs a CUDA toolkit to JIT its kernels (KI-006); hqq was a drop-in replacement with the same cache semantics.

## Limitations

- One checkpoint (Falcon-E 1B), one quantization recipe per component, no calibration-based head quantizer (GPTQ/AWQ-style) for comparison, no INT8 fallbacks needed or run.
- PILOT scale for KV quality: 3,840–3,968 scored tokens per context (1.1% of WikiText-2 test), one or two windows; HellaSwag 2,000 of 10,042. No stderr on the incremental perplexity (per-window NLLs are in the JSON: two windows at 2k, one at 4k).
- Context above 4k was tested for memory only, not quality; 32k not at all. Generation checks are single greedy samples.
- Latency is eager PyTorch under WSL2 on a laptop GPU with the ternary weights unpacked to BF16 (R7); absolute numbers mean nothing outside relative comparison between configurations in the same mode.
- The quantized cache dequantizes the whole history every step: this experiment establishes numerical feasibility of the representation, not traffic savings.
- lm-eval tasks do not exercise the KV cache (R2 control); all KV quality evidence rests on R3 and R4.
- The head quantizer's max error on the real weights is 1.007 × scale/2 in float32 (fp16 scale rounding; the unit test allows 2% slack on synthetic weights and passes) and 1.073 × scale/2 when the reconstruction is rounded to BF16, the form used at execution (`head_int4.json` reports the BF16 figure; 28% of groups exceed half a step by BF16 rounding alone). Checked after the run on the CPU; not part of the committed outputs.
- Long GPU runs under the WSL memory cap were killed twice by host memory pressure (KI-007); the driver was made resumable and the measurements were produced in chunks, which is why `manifest_*.json` timestamps differ per phase.

## What This Proves

- MEASURED: on this checkpoint, an INT4 RTN g128 output head (4.16 bits/element) changes WikiText-2 perplexity by +0.67% and no benchmark accuracy beyond one standard error.
- MEASURED: a 4-bit KIVI-style KV cache (4.5 bits/element effective) exercised through real prefill, residual accumulation, 15–31 re-quantization flushes and per-step retrieval changes teacher-forced perplexity by +0.17% to +1.07% at 2k–4k context, with per-token grouping at the low end; resident cache memory is 3.5× smaller.
- MEASURED: the two quantizations combine additively at 2k context.
- MEASURED: the reference model's KV cache is BF16 `[1, 2, T, 128]` per layer and its head is an untied BF16 `[32768, 2048]` linear, confirming EXP-0003's inputs.
- MEASURED: on this runtime neither representation reduces bytes read during attention, and the quantized cache slows prefill 4–8×.

## What This Does NOT Prove

- Any hardware result, energy or throughput benefit; any claim relative to a GPU or any competitor.
- That these quality costs hold at longer contexts, on other checkpoints (notably 128k-vocabulary heads), with sampling instead of greedy decoding, or on tasks that stress long-range recall.
- That the candidate format is FlexTern's format; it is a candidate for a founder decision and an ADR, after review.
- That the EXP-0003 "logical" ceilings are achievable: they require an attention path that reads 4-bit K/V directly, which exists on GPUs as fused kernels and does not exist in FlexTern.

## Artifacts / Logs

`results/model_precision/`: `environment.json`, `audit.json`, `lm_eval*.json`, `incremental_*.json`, `generation.json`, `memory.json`, `latency.json`, `head_int4.json`, `manifest_*.json`, `quality_comparison.csv`, `memory_comparison.csv`, `latency_comparison.csv`, `candidate_format.md`, `roofline_reconnect.{csv,md}`. Code: `analysis/model_precision/`, `tests/test_model_precision.py`, `scripts/run_model_precision.ps1`.

## Follow-up Questions

- Does a 128k-vocabulary tied head (BitNet 2B4T) tolerate the same INT4 recipe, and does a calibration-based quantizer (GPTQ/AWQ-style) buy anything over RTN on these models?
- At what context (8k–32k) does 4-bit per-token KV quality degrade, and is the residual window the right lever?
- Does the earlier greedy-loop entry under quantization persist under sampling or repetition penalties, or is it a greedy artifact only?
- Would 2-bit or 3-bit KV with the same structure still be acceptable, given the 4.5-bit effective cost of metadata at g = 64?
- Can a FlexTern attention path consume 4-bit K/V with per-group scale and zero directly (Q-022), and what does that engine cost?
