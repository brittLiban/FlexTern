# ADR-0004 — Reference workload set: BitNet b1.58 2B4T plus Falcon-E 1B/3B

Status: Accepted
Date: 2026-10-06

## Context

ADR-0003 fixed a single reference workload (Microsoft BitNet b1.58 2B4T) and stated that findings are model-specific until a second model confirms them. EXP-0002 produced hardware-relevant conclusions from that one checkpoint. Before any Youssouf Architecture decision hardens around its dimensions, an independent native ternary family is needed to separate what generalizes from what is Microsoft-specific.

## Decision

The reference workload set is now three pinned checkpoints, all profiled with the same tool (`analysis/ternary_model_profile.py`, specs in `analysis/model_specs.py`):

| Slug | Model | Revision | File | Bytes | SHA-256 | License |
|---|---|---|---|---:|---|---|
| `bitnet-2b4t` | `microsoft/bitnet-b1.58-2B-4T` | `04c3b9ad…` | `model.safetensors` | 1,178,623,988 | `8143ae11…975b` | MIT |
| `falcon-e-1b` | `tiiuae/Falcon-E-1B-Base` (branch `main`) | `f4001b8b…` | `model.safetensors` | 665,041,488 | `f62c270b…c79d` | Falcon-LLM License (TII) |
| `falcon-e-3b` | `tiiuae/Falcon-E-3B-Base` (branch `main`) | `ad18b071…` | `model.safetensors` | 998,568,704 | `25dbd87d…6ba2` | Falcon-LLM License (TII) |

Latent checkpoints used only for cross-checks: `microsoft/bitnet-b1.58-2B-4T-bf16` @ `27668139…` and `tiiuae/Falcon-E-1B-Base` branch `prequantized` @ `ff68bcdd…`. Falcon's `bfloat16` branch is a different, non-ternary model and is not used.

Reference code: `huggingface/transformers` `integrations/bitnet.py` @ `d6abd133` (both families) and `tiiuae/onebitllms` @ `9be8e4f3` (Falcon training kernels and `quantize_to_1bit`).

Any cross-model claim in FlexTern documentation cites this set. A property is called "likely generalizing" only when it holds across both families; two families are evidence, not proof.

## Alternatives Considered

- **Keep one reference model.** Rejected: EXP-0002B shows several EXP-0002 findings (dead channels, head dominance, exact K, tied head) were Microsoft-specific.
- **Falcon-E Instruct variants.** Rejected: same architecture and quantization as Base; Base avoids fine-tuning artifacts.
- **Config-only inspection of 3B.** Rejected: the generalized profiler made a full 3B run cheap.
- **Post-training-quantized ternary conversions of other models.** Rejected: not native ternary training.

## Why

Falcon-E is the only other publicly released, natively trained ternary family at comparable scale with a published reference implementation. It differs from BitNet in organization, architecture (Llama/SwiGLU, no sub-norms, untied head, 32k vocabulary, 32k context, GQA 8:1), training kernels, and checkpoint scale convention, which makes agreement between the two meaningful.

## Consequences

- EXP-0002's canonical Microsoft outputs stay in `results/workload_profile/` unchanged (verified: 10,929 numeric values identical after the refactor). Falcon outputs live in `results/workload_profile/falcon-e-{1b,3b}/`; the comparison in `results/workload_profile/cross_model/`.
- The profiler is spec-driven; adding a transformers-packed BitNet-format model is a spec entry, not a code change.
- Models not in this set, or with different packing, require a new ADR before their numbers are used.

## Evidence

Founder direction, 2026-10-06 (EXP-0002B prompt). Provenance verified via the Hugging Face API, headers, model cards, the Falcon-Edge blog, and the pinned reference code.

## Supersedes

ADR-0003.

## Superseded By

None.
