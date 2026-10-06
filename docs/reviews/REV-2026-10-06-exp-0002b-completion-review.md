# Review — EXP-0002B cross-model validation (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: EXP-0002B and its artifacts (`analysis/`, `tests/test_analysis_cross_model.py`, `results/workload_profile/{falcon-e-1b,falcon-e-3b,cross_model}/`, the regenerated BitNet outputs, ADR-0004, updated canonical docs, the handoff), as they stood immediately before the EXP-0002B commit. The founder asked for attacks on eight specific points; each has a verdict below.

## Overall Assessment

The objective is complete. The tooling is sound, the regression gate is real (10,929 BitNet numeric values identical), Falcon-E provenance is stronger than BitNet's, and the invariant classification is appropriately conservative once one qualifier is applied: Falcon-E shares BitNet's quantization recipe by deliberate adoption, so recipe-level invariants are one recipe reproduced by two labs, not independent convergence. That qualifier was added to EXP-0002B during this review. No number in EXP-0002B disagrees with the generated comparison (checked: MLP shares 76.4/85.7/89.7%, KV bytes at both contexts, head percentages, accumulator bits at all five K, ±balance deltas).

## What Looks Strong

- Provenance: pinned revisions, etag SHA-256s, header-derived tensor counts, model cards, blog, and pinned reference code at two repositories. The `bfloat16`-branch trap (a different non-ternary model) was correctly avoided.
- Falcon-E's packed branch reproduces bit-for-bit from its `prequantized` branch under BF16 arithmetic emulation. That is stronger checkpoint provenance than BitNet has (KI-004).
- The scale-convention difference (β vs 1/β) was derived from the transformers dispatch and confirmed on data, then correctly classified as a loader concern.
- Spec-driven profiler; the BitNet default output location is unchanged; pinning tests exist for both families.

## Concerns

- **Independence (attack 1).** Independent in organization, run, data, architecture (Llama/SwiGLU, no sub-norms, untied head, 32k vocab/context, GQA 8:1), and training kernels. Not independent in recipe. Verdict: valid second family for architecture-level and sparsity-level properties; recipe-level properties (ternary set, absmean, INT8 per-token absmax, per-tensor scale) are "likely generalizing among BitNet-recipe models." Applied.
- **Provenance (attack 2).** Verified as above. One unresolved item recorded: the model card's "1.8B" versus the shape-derived 1,719,765,160 for Falcon-E 1B.
- **Native ternary claim (attack 3).** Justified: single training run yielding the quantized model (blog), TII's own STE Triton kernels, and the exact reproduction of the packed branch from the latent branch.
- **Shared semantics (attack 4).** Comparable at the linear-layer and storage level (identical packing, naming, scale granularity). Not comparable in the non-linear op mix (ReLU² + sub-norms vs SiLU); the per-model inventories handle that correctly and the comparison does not average across it.
- **256-wide tiling (attack 5).** Not over-interpreted: MEASURED divisibility on five values, GCD computed, 6912 shown to block 512, optimality explicitly disclaimed. Added during review: both hidden sizes are multiples of 512; only BitNet's intermediate size forces 256.
- **Activation precision (attack 6).** Correctly scoped as current cross-model evidence, with Q-021 tracking the sub-8-bit risk. With the recipe qualifier it is appropriately weak.
- **Sparsity differences (attack 7).** They do not invalidate prior hardware thinking; they firm it up. EXP-0002 already declined per-element skipping; Falcon removing the only structured sparsity makes Q-020 compiler-only, if ever.
- **Roofline parameterization (attack 8).** Adequate once batch size and prefill-vs-decode were added to the envelope (done during review). Reading inputs from the three `model_summary.json` files is the right discipline.

## Assumptions Being Made

- That Falcon-E 3B shares 1B's scale convention. Inferred from the shared code path and stored-scale magnitudes (5–58), not cross-checked on latents (sharded branch).
- That static MAC and byte counts remain a usable proxy absent runtime data.

## Potential Failure Modes

- Reading "likely generalizing" as universal. Three checkpoints, two families, one recipe.
- Treating the KV-cache finding (805–1,074 MB/token at 32k context) as a measurement; it is the same stream-once analytical model.

## Benchmark / Scientific Concerns

- The original plan review contained an arithmetic slip (3B ternary total given as the MLP-only sum). The implementing agent's shape-derived assert caught it; the correction is now recorded in that review rather than silently overwritten.

## Architecture Questions

- Whether a FlexTern loader should canonicalize (code assignment, scale convention) once, given both checkpoints use codes Y-Core v0.1 does not, and the two families store the scale inversely.

## Alternative Approaches Worth Considering

- A third family, if one appears, before any RTL parameter is frozen (Q-018).

## Questions for Founder / ChatGPT

- Q-019: head (BitNet) vs KV cache at long context (Falcon) as the primary non-ternary stream for the roofline.
- Q-021: activation bits as a first-class parameter from the roofline onward.

## Recommended Actions

1. Commit as `research: validate FlexTern workload assumptions across Falcon-E`, then a docs-only hash fixup; push under this task's authorization after staged-file inspection.
2. Proceed to the roofline with the EXP-0002B envelope as its input space.

## Severity

Informational. No defects found; the independence qualifier and envelope addition were the only substantive changes, both applied before commit.
