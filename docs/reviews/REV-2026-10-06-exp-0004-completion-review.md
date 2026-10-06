# Review — EXP-0004 low-precision head / KV feasibility (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: EXP-0004 and its artifacts (`analysis/model_precision/`, `tests/test_model_precision.py`, `results/model_precision/`, ADR-0006, the EXP-0004 record, the handoff, and the THESIS / OPEN_QUESTIONS / PROJECT_STATE wording) as they stood immediately before the EXP-0004 commit. Per the founder's instruction that the reviewer receive a different prompt from the implementation, the following twelve points were put to the reviewer verbatim: attack (1) benchmark validity, (2) the perplexity calculation, (3) dataset selection, (4) sample sizes, (5) quantization correctness, (6) KV append / cache semantics, (7) dequantization overhead, (8) memory accounting, (9) misleading runtime comparisons, (10) quality claims, (11) INT4 baseline fairness, (12) hardware implications and the recommended next objective.

## Overall Assessment

Every number in the record recomputes from the JSON (R2 deltas, R3 percentages, R5 bytes, ratios and bits, R6 medians, R8 ceilings, the 4.5-bit metadata arithmetic, the strided-group diagnosis against the hqq `[32, 1348]` layout). Two items blocked the commit as authorized and were resolved before it: the three model-level tests had only ever been skipped (now executed with the model loaded: 11 of 11 pass in 71 s), and the commit trailer had changed mid-session. Five wording defects were corrected: a resident-memory delta that quoted the computed head shrink (99 MB) instead of the measured value (91 MB); the 4k incremental window being the same text as the two 2k windows (context length is tested on identical text, not on more data); a divergence point in a generation sample (about 15 tokens, not eight); an apparent contradiction between B's lower decode-step peak and R8's "attainable" row (peak residency versus bytes moved); and the head error bound, which is 1.007 × scale/2 in float32 and 1.073 × scale/2 only after BF16 rounding of the reconstruction. The handoff's claim that EXP-0004 "removed" the model-side objection was softened to "weakened on one checkpoint at pilot scale". No headline number changed.

## What Looks Strong

- The KV path is exercised for real: lazy prefill quantization, 128-token residual, whole-history re-quantization, per-step retrieval, through 15 (2k) and about 31 (4k) flushes, with per-position NLL buckets to detect drift.
- The lm-eval invariance trap was demonstrated, not assumed (B equals baseline to every digit), and no lm-eval number is presented as KV evidence.
- The grouping-axis finding is diagnosed from source (`reshape(group_size, -1)` stripes groups across tokens, so every flush re-groups old tokens), confirmed by a synthetic drift test (0.35 vs 7e-7), and confirmed again in the real evaluation (+1.07% vs +0.34% at 4k).
- Logical, resident-as-stored, temporary and peak bytes are kept apart, and peaks disclose the 3.17 GB of ternary weights unpacked to BF16.
- The baseline was sampled on the long prompt too, which prevented the quote-loop from being misattributed to quantization.

## Concerns (verdicts on the twelve points)

1. **Benchmark validity.** Sound. Deterministic loglikelihood scoring and greedy teacher-forcing, pinned settings (max_length 2048, batch 4, seed), baseline from the same harness. No external anchor exists for these tasks on this model; stated.
2. **Perplexity calculation.** Two distinct quantities: lm-eval rolling word perplexity at 2048 (R2) and token perplexity over positions ≥ 128 after a 128-token prefill (R3). Both valid; now named distinctly. The incremental baseline sits 0.13–0.14% below the single-forward reference in the same direction at both contexts, a small systematic sdpa decode-path difference, inside tolerance and now noted as systematic rather than noise.
3. **Dataset selection.** WikiText-2 for perplexity and for generation prompts. In-distribution for a Wikipedia-trained model; the quote loop is a WikiText-tokenization attractor. Fine for relative comparison and caveated. No held-out long-recall task; listed under what is not proven.
4. **Sample sizes.** PILOT labels present (HellaSwag 2,000 of 10,042; 1.1% of WikiText-2 test for the incremental runs). The 4k window overlaps the 2k windows exactly; now stated. No stderr on R3; per-window NLLs exist for the two 2k windows only.
5. **Quantization correctness.** Head: standard asymmetric RTN with explicit packing; round-trip, exact-code and matmul tests pass; the fp16-scale and BF16-rounding overshoot of the half-step bound is documented with both figures. KV: hqq 4-bit g64 with BF16 scale and zero; axis-0 strided versus axis-1 contiguous groups correctly characterized.
6. **KV semantics.** Measured from the transformers 5.18.0 source and exercised. The one-token residual after the sweep's single decode step reconciles exactly (24,576 B = 24 layers × K,V × 2 heads × 128 × 2 B). Correct.
7. **Dequantization overhead.** Head: +0.5 GB decode-step peak despite −91 MB resident, reported. KV: per-layer BF16 temporary, freed per layer, so peak residency falls while bytes moved per token do not; reconciled in R5.
8. **Memory accounting.** Separated and labeled; the 99 → 91 MB correction made; the unexplained 8 MB gap and the INFERRED attribution of B's +44 MB are stated.
9. **Runtime comparisons.** Cached-unpack mode disclosed, lm-eval in stock mode disclosed, identical logits shown. Quantized cache: prefill 4–8× slower, decode 11–15% slower; relative-only framing is correct and the record says absolute numbers mean nothing outside it.
10. **Quality-claim wording.** THESIS, OPEN_QUESTIONS and PROJECT_STATE carry the 0.17–1.07% range, the per-token-grouping qualifier and the fact that baseline also loops. "Tolerates" is interpretive, but every instance carries MEASURED / one checkpoint / pilot scale.
11. **INT4 fairness, categories A/B/C.** Both representations placed in category A (any GPU can use them today). The record says the right comparator for any FlexTern claim is a GPU already using an INT4 head and 4-bit KV, and cites EXP-0003's 1.3–1.8× ternary-stream-only byte advantage against such a system. R8 "logical" rows are labeled as ceilings; no competitor number is claimed.
12. **Hardware implications and next objective.** Follow from the evidence with the stated caveat that attainable equals the INT4-head rows until an engine consumes 4-bit operands directly (Q-022). The single recommendation (the streaming ternary engine study from the EXP-0003 handoff, with the EXP-0004 candidate format as a fixed interface assumption) is defensible; its wording was softened.

## Assumptions Being Made

- One checkpoint, one recipe per component, greedy decoding, ≤ 4k context for quality; transfer to 128k-vocabulary tied heads and to long-recall tasks is untested.
- The hardware consumer of either representation is hypothetical; GPU fused kernels for both exist elsewhere and were not measured.

## Potential Failure Modes

- Quoting the R8 "logical" ceilings as FlexTern speedups.
- Reading "resident cache 3.5× smaller" as "attention traffic 3.5× smaller".
- Treating the earlier quote-loop entry under quantization as a measured quality loss (single greedy sample) or ignoring it entirely (it is a brittleness signal worth a sampling-based follow-up).
- Adopting transformers' default KV grouping axis in any future FlexTern format; it is the drifting one.

## Recommended Changes

All applied before commit: run the gated model-level tests and report real counts; the five wording corrections above; the handoff softening; the two-line commit trailer.
