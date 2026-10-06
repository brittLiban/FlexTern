# Review — EXP-0002B cross-model validation plan (pre-implementation)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: The plan to profile TII Falcon-E 1B/3B with the generalized Phase 2 profiler, before any code was changed. Inputs reviewed: Hugging Face API metadata and safetensors headers for `tiiuae/Falcon-E-1B-Base` (main f4001b8b, prequantized ff68bcdd, bfloat16 d8c43c84) and `tiiuae/Falcon-E-3B-Base` (main ad18b071); `config.json` of each; the Falcon-Edge blog; `tiiuae/onebitllms` source at commit 9be8e4f3; `transformers/integrations/bitnet.py` at d6abd133.

## Overall Assessment

Falcon-E is a genuine fit and genuinely independent in the ways that matter: a different organization, a different architecture family (Llama/SwiGLU, no sub-norms, untied head, 32k vocabulary, GQA 8:1, 32k context), TII's own Triton training kernels. What it shares with Microsoft is the transformers packing format and the inference-time activation quantizer, both by construction. No escalation needed.

## What Looks Strong

- Header-only inspection established shapes, dtypes, tensor counts (387 and 515) and file sizes before any download.
- Scale convention was derived from code: no `linear_class` in the Falcon config routes to `BitLinear`, which divides by `weight_scale`; `onebitllms.quantize_to_1bit` stores `1/mean|W|`. Microsoft's `AutoBitLinear` offline mode multiplies and stores `mean|W|`. Opposite checkpoint conventions, identical hardware meaning.

## Concerns

- The three Falcon branches are not interchangeable. `prequantized` is the latent checkpoint that `quantize_to_1bit` consumes; `bfloat16` is a different, non-ternary model from the same run. Only `prequantized` is valid for a ternarization cross-check.
- Microsoft-specific literals are embedded in the profiler beyond the regexes: the sub-norm regex and the `4L+1` assert, the "no lm_head" assert, a hardcoded "55.7% of file" note, "210 tensors" and "multiplies the output" prose, ReLU² and sub-norm op rows, and a traffic model that equates head bytes with embedding bytes (wrong when untied).
- The regression gate must be real: after refactoring, the Microsoft CSV/JSON outputs must be byte-identical.

## Assumptions Being Made

- That Falcon's packed `main` branch was produced by `quantize_to_1bit` from `prequantized`. Agreement may then be 100% rather than Microsoft's 98–99%; verify rather than assume.
- That 3B shares the 1B tensor naming. Confirmed from the header.

## Potential Failure Modes

- Over-reading "all observed K divisible by 256" as a design law. Five observed K values (2048, 2560, 6912, 9216, 13312) all divide by 256; 6912 does not divide by 512. That is a MEASURED fact about two families, nothing more.
- Overstating activation independence: both inference paths run transformers' quantizer. The independent evidence is TII's training kernel (`127/absmax`, `floor(x+0.5)`, clamp [-128,127]), which reaches the same [-127,127] range by a different rounding rule.

## Benchmark / Scientific Concerns

- Rounding differs between TII's kernel (half-up) and `quantize_to_1bit` (torch half-even). Record it; it does not change ranges.
- Report KV traffic at 4096 and at each model's maximum context, since Falcon's is 32k.

## Architecture Questions

- Whether K = 2048 is the first full-length case where "-128 unreachable" saves an accumulator bit (128 × 2048 = 2^18 exactly). Check.
- Head share: ~20% of file for Falcon 1B versus 56% for Microsoft. If so, head precision generalizes but head dominance was vocabulary-driven.

## Alternative Approaches Worth Considering

- Config-only validation of 3B. Rejected: the refactored profiler makes a full 3B run cheap (1 GB download, ~1 minute).

## Questions for Founder / ChatGPT

- None blocking. The completion review should return to the independence question.

## Recommended Actions

1. Model spec table with `scale_convention`, norm list, tied flag, prose fields, default output directory; Microsoft default output unchanged.
2. Generalized reconciliation asserts (tensor count formula, packed rows × 4 = config N using `head_dim`, expected ternary totals 1,585,446,912 and 2,919,235,584 (the 3B figure corrected from 2,617,245,696, an arithmetic slip in the original review that counted only the MLP projections; caught by the implementing agent's shape-derived assert), file-size reconciliation).
3. Regression: rerun Microsoft offline, require empty `git diff` on CSV/JSON; add a test pinning the committed module totals.
4. Cross-check on `prequantized`; report `mean|W| × stored_scale ≈ 1` for Falcon.
5. ADR-0004 superseding ADR-0003 with the reference workload set; Q-018 partially answered; new Q-021 on parameterized activation precision.
6. Persist this review and a completion review.

## Severity

Informational. No defects; the concerns are about what to verify and how to label.
