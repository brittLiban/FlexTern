# Review — EXP-0004 low-precision head / KV feasibility plan (pre-implementation)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: The plan before any quantization code was written. Inputs reviewed: the EXP-0004 prompt, hardware audit (RTX 5060 Laptop 8 GB, 15 GB RAM, no MSVC), transformers v5.18.0 `cache_utils.py` (`QuantizedLayer`, `QuantoQuantizedLayer`, `QuantizedCache`) and `integrations/bitnet.py`, package availability (lm_eval 0.4.13, optimum-quanto 0.2.7, hqq 0.2.8).

## Overall Assessment

Design is sound: a custom auditable round-to-nearest groupwise INT4 head with explicit packing, the transformers KIVI-style quantized cache for KV, lm-eval-harness for head configurations, and teacher-forced incremental decode through the real cache for KV configurations. No paid compute is needed.

## What Looks Strong

- The KV path exercises real prefill, lazy cache initialization, residual accumulation, re-quantization, and retrieval, not a one-shot tensor quantization.
- The head quantizer is small enough to test exactly (pack/unpack round trip, error bound of scale/2, matmul against reference).

## Concerns

- **lm_eval invariance trap.** Loglikelihood tasks run one forward with no incremental cache, so KV configurations score identically to their head-equivalents by construction. Demonstrate once, state it, and never present those scores as KV evidence.
- **Re-quantization in `QuantizedLayer.update`.** Every time the residual fills, the implementation re-quantizes `cat([dequantized old, residual, new])`. Whether that compounds error depends on the grouping axis. Test on a synthetic tensor and report NLL per position bucket in the real evaluation.
- **Whole-cache dequantization every step.** Resident cache memory shrinks; external traffic on this runtime does not. Report both outcomes separately.
- **Deserialization path.** `BitNetDeserialize` can unpack ternary weights to BF16 in some flows; assert `q_proj.weight` is uint8 [512, 2048] after load or resident memory silently inflates five-fold.
- **Install ordering.** Do not install lm_eval/quanto/hqq until the cu128 torch job finishes, or pip may resolve a CPU torch. Set `TORCHDYNAMO_DISABLE=1` everywhere (no MSVC for inductor). transformers 5 uses `dtype=` not `torch_dtype=`.

## Assumptions Being Made

- lm_eval 0.4.13 works with transformers 5.18; fall back to the newest supported 4.5x or self-scored tasks if it does not.
- quanto falls back to pure torch on Windows if its extension does not build; otherwise use the hqq backend.

## Potential Failure Modes

- Reporting theoretical INT4 bytes as measured savings; reporting eager Windows latency as representative; presenting a handful of prompts as quality evidence.

## Benchmark / Scientific Concerns

- Baseline incremental decode with `DynamicCache` must reproduce single-forward perplexity on the same windows within BF16 tolerance; that validates positions and causal masking before any quantized number is trusted.
- Fix WikiText-2 windows up front; identical offsets for all configurations; label subsets PILOT.
- Compare baseline lm_eval scores with the model card as an external anchor only.

## Architecture Questions

- Effective bits per KV element including scale, zero, and the 128-token BF16 residual.

## Alternative Approaches Worth Considering

- HQQ 4-bit head as a second frontier point; not required.

## Questions for Founder / ChatGPT

- None blocking.

## Recommended Actions

1. ADR-0006 recording the model-side co-design track; H6 in THESIS as HYPOTHESIZED; ROADMAP note.
2. Smoke audit script asserting checkpoint SHA-256, uint8 ternary weights, BF16 `lm_head` [32768, 2048], default cache dtype/shape, and quantized-cache internals after a 300-token generation.
3. Report logical bytes, resident bytes, and peak allocation separately; feed measured effective bytes per element into a copy of the EXP-0003 scenarios without touching `results/roofline/`.
4. Classify both quantizations under category A (available to any hardware); mark B/C unresolved.
5. Persist this review.

## Severity

Informational.
