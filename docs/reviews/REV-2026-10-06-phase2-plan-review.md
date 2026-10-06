# Review — Phase 2 workload-characterization plan (pre-implementation)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: The plan for EXP-0002 before any analysis code was written. Inputs reviewed: the founder's Phase 2 prompt, Hugging Face API metadata for `microsoft/bitnet-b1.58-2B-4T` (revision 04c3b9ad), the safetensors header (542 tensors), `config.json`, and the transformers `integrations/bitnet.py` source.

## Overall Assessment

Model selection is sound: the preferred model is ungated, MIT-licensed, 1.18 GB, and natively ternary with packed 2-bit storage. No escalation needed. The plan is executable without loading the full model into memory.

## What Looks Strong

- Header-only inspection already established file layout, dtypes, and shapes before any download.
- The analysis can memory-map the file and never touch the 657 MB embedding bytes.
- Packing semantics are available verbatim from the reference implementation, with a docstring example usable as a test vector.

## Concerns

- `config.json` names `linear_class: autobitlinear`, but only `BitLinear.forward` had been read verbatim. The two classes differ in whether the stored scale multiplies or divides the output, which decides whether `weight_scale` is β or 1/β. Resolve from code and confirm from data.
- The tied BF16 output head (128,256 × 2,560) may be the single largest per-token weight stream in batch-1 decode, larger than all packed ternary layers combined. This must not be buried under ternary statistics.
- Activation quantization uses scale = 127/absmax, so -128 may be unreachable. If so, Y-Core's 9-bit output covers a case this model never produces. Report; do not modify RTL.

## Assumptions Being Made

- That the packed checkpoint's 2-bit codes contain no value 3. Count and report it rather than assume.
- That only two dot lengths (2,560 and 6,912) exist. Verify from shapes.

## Potential Failure Modes

- Silent tensor skipping. Mitigate with a byte-range reconciliation: 8 + header length + sum of tensor byte ranges must equal the file size.
- Hand-transcribed numbers drifting from generated ones. Mitigate by having the tool emit the summary markdown.

## Benchmark / Scientific Concerns

- All arithmetic and traffic figures are INFERRED; label them. No runtime is measured.
- Report accumulator widths for both the INT8 format range and the model-actual range.

## Architecture Questions

- Whether the output head, not the ternary datapath, dominates decode weight traffic. This is the question most likely to change the project's next step.

## Alternative Approaches Worth Considering

- HTTP range streaming instead of a full download. Not needed at 1.18 GB with 636 GB free.

## Questions for Founder / ChatGPT

- If the output head dominates traffic, is it in scope for FlexTern or an accepted fixed cost?

## Recommended Actions

1. Fetch `AutoBitLinear.forward` verbatim and pin the transformers commit.
2. Add reconciliation asserts (byte ranges, tensor count 542 = 2 + 30 × 18, packed rows × 4 = config dims).
3. Brute-force validate the accumulator-width formula at K = 1, 2.
4. Measure aligned-group all-zero rates along K against the independence prediction.
5. Record this review and a completion review under `docs/reviews/`.
6. Fix "adder tree" → "adder chain" wording while touching docs.
7. Rank Part 13 candidates from the data, even if it contradicts the Phase 1 handoff.

## Severity

Informational. No blocking issues; the concerns above are about what to verify, not about a flaw in the plan.
