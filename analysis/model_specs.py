"""
Per-model specifications for the FlexTern ternary workload profiler.

Everything that differs between supported checkpoints lives here, so the
profiler itself contains no model-specific constants. Adding a model that
uses the transformers BitNet packing (uint8, 4 ternary codes per byte along
the output dimension, one BF16 weight_scale per tensor, Llama-style tensor
names) should only require a new entry.

Scale convention (verified from reference code, see each entry):
    "multiply": stored weight_scale = beta = mean|W_latent|; output = linear(x, W_q) * weight_scale
                (transformers AutoBitLinear, offline mode)
    "divide":   stored weight_scale = 1/beta;                output = linear(x, W_q) / weight_scale
                (transformers BitLinear; onebitllms quantize_to_1bit stores 1/mean|W|)
Either way the effective weight is beta * W_q and hardware sees one scalar
multiply per output element after accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    family: str
    model_id: str
    revision: str
    expected_file_size: int
    expected_sha256: str | None
    license: str
    paper: str
    code_source: str
    native_ternary_claim: str
    scale_convention: str                 # "multiply" | "divide"
    norm_names: tuple[str, ...]           # per-layer norm tensor stems under model.layers.N.
    tied_head: bool
    default_out: str
    act_quant_source: str                 # where the activation quantizer was read from
    act_rounding: str                     # rounding rule of the activation quantizer
    weight_rounding: str                  # rounding rule used to produce the checkpoint
    prose_normalization: str
    prose_attention: str
    prose_mlp: str
    latent_repo: str | None = None        # repo holding BF16 latent weights for cross-check
    latent_revision: str | None = None
    latent_note: str = ""
    crosscheck_tensors: tuple[str, ...] = field(default_factory=tuple)
    config_file: str = "config.json"
    weights_file: str = "model.safetensors"


# Activation quantizer facts shared by both families (verified separately per family):
# per-token absmax, scale = 127/absmax, clamp to [-128, 127]. Because |x*scale| <= 127
# before rounding, -128 is unreachable under either rounding rule.
ACT_BITS = 8
ACT_FORMAT_MIN, ACT_FORMAT_MAX = -128, 127
ACT_MODEL_MIN, ACT_MODEL_MAX = -127, 127

TRANSFORMERS_BITNET_COMMIT = "d6abd1333078195535f461589b64f112b53688b2"  # integrations/bitnet.py
ONEBITLLMS_COMMIT = "9be8e4f377bd7fa0e86fc0e3083f7785dc805201"

SPECS: dict[str, ModelSpec] = {
    "bitnet-2b4t": ModelSpec(
        slug="bitnet-2b4t",
        family="Microsoft BitNet b1.58",
        model_id="microsoft/bitnet-b1.58-2B-4T",
        revision="04c3b9ad9361b824064a1f25ea60a8be9599b127",
        expected_file_size=1_178_623_988,
        expected_sha256="8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b",
        license="MIT (LICENSE file in model repo; Hugging Face tag license:mit)",
        paper="BitNet b1.58 2B4T Technical Report, arXiv:2504.12285",
        code_source=f"huggingface/transformers src/transformers/integrations/bitnet.py @ {TRANSFORMERS_BITNET_COMMIT} "
                    "(AutoBitLinear, offline mode; config linear_class=autobitlinear)",
        native_ternary_claim="Model card: 'trained from scratch with this quantization scheme, not post-training quantized'; "
                             "4T tokens; absmean ternary weights, 8-bit absmax per-token activations (W1.58A8).",
        scale_convention="multiply",
        norm_names=("input_layernorm", "post_attention_layernorm", "self_attn.attn_sub_norm", "mlp.ffn_sub_norm"),
        tied_head=True,
        default_out="results/workload_profile",
        act_quant_source="transformers AutoBitLinear/ActQuant (inference) — same code as BitLinear.activation_quant",
        act_rounding="torch round (half-to-even), clamp [-128,127]",
        weight_rounding="torch round (half-to-even) in WeightQuant; packed repo produced by Microsoft (method confirmed on latents, 98-99.4% code agreement)",
        prose_normalization="RMSNorm (BitNetRMSNorm) x4 per layer: input_layernorm, post_attention_layernorm, "
                            "self_attn.attn_sub_norm (before o_proj), mlp.ffn_sub_norm (before down_proj); plus final model.norm",
        prose_attention="GQA, RoPE on q/k, attn_sub_norm applied to attention output before o_proj, no bias",
        prose_mlp="down_proj(ffn_sub_norm(relu2(gate_proj(x)) * up_proj(x)))",
        latent_repo="microsoft/bitnet-b1.58-2B-4T-bf16",
        latent_revision="276681394656abdadb8e80e5b2c3db5e5d7fcaff",
        latent_note="BF16 master weights published for fine-tuning; not guaranteed to be the exact snapshot the packed repo was made from.",
        crosscheck_tensors=("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
                            "model.layers.17.mlp.down_proj.weight"),
    ),
    "falcon-e-1b": ModelSpec(
        slug="falcon-e-1b",
        family="TII Falcon-E (Falcon-Edge)",
        model_id="tiiuae/Falcon-E-1B-Base",
        revision="f4001b8b1c26d28a717d79a8ece14901816d92e8",   # branch main = packed BitNet checkpoint
        expected_file_size=665_041_488,
        expected_sha256="f62c270b5640c9fce1dedf1ffe84aa8f138443ad1b065e90c7508a0eebc7c79d",
        license="Falcon-LLM License (TII), Apache-2.0-based with Acceptable Use Policy; Hugging Face tag license:other; "
                "https://falconllm.tii.ae/falcon-terms-and-conditions.html",
        paper="Falcon-Edge release blog, https://falcon-lm.github.io/blog/falcon-edge/ (no arXiv report at time of analysis)",
        code_source=f"huggingface/transformers integrations/bitnet.py @ {TRANSFORMERS_BITNET_COMMIT} (BitLinear; config has "
                    f"quant_method=bitnet and no linear_class) + tiiuae/onebitllms @ {ONEBITLLMS_COMMIT} "
                    "(kernels/activation_quant.py, kernels/weight_quant.py, utils/quantization_utils.py)",
        native_ternary_claim="Blog: pre-training 'simultaneously yields both non-quantized and quantized model variants'; ~1.5T tokens; "
                             "BitNet b1.58 architecture with BitNet-internal layer norms removed; weights absmean-ternarized and "
                             "activations int8 per-token absmax in TII's Triton training kernels.",
        scale_convention="divide",
        norm_names=("input_layernorm", "post_attention_layernorm"),
        tied_head=False,
        default_out="results/workload_profile/falcon-e-1b",
        act_quant_source="onebitllms kernels/activation_quant.py (training): scale = 127/row_absmax (min 1e-5), floor(x*scale+0.5), "
                         "clamp [-128,127]; transformers BitLinear.activation_quant (inference)",
        act_rounding="training kernel floor(x+0.5) (half-up); inference torch round (half-to-even); clamp [-128,127]",
        weight_rounding="onebitllms quantize_to_1bit: torch round (half-to-even); training kernel floor(x+0.5)",
        prose_normalization="RMSNorm (LlamaRMSNorm) x2 per layer: input_layernorm, post_attention_layernorm; plus final model.norm. "
                            "No sub-norms (BitNet-internal norms removed per blog).",
        prose_attention="Llama GQA, RoPE on q/k, no bias",
        prose_mlp="down_proj(silu(gate_proj(x)) * up_proj(x))  (SwiGLU)",
        latent_repo="tiiuae/Falcon-E-1B-Base",
        latent_revision="ff68bcdd17933f7cf06a147360018c2f60e57be8",   # branch prequantized = latent weights consumed by quantize_to_1bit
        latent_note="'prequantized' branch: BF16 latent weights for fine-tuning, the input of onebitllms.quantize_to_1bit. "
                    "The separate 'bfloat16' branch is a different non-ternary model from the same run and is NOT used.",
        crosscheck_tensors=("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
                            "model.layers.12.mlp.down_proj.weight"),
    ),
    "falcon-e-3b": ModelSpec(
        slug="falcon-e-3b",
        family="TII Falcon-E (Falcon-Edge)",
        model_id="tiiuae/Falcon-E-3B-Base",
        revision="ad18b0713c10a8696144b1112accf4ccd28af6d3",
        expected_file_size=998_568_704,
        expected_sha256="25dbd87d68cf853de3bea1ae4615bf2fa027f252b77ec2137a07e94c1a1a6ba2",
        license="Falcon-LLM License (TII), Apache-2.0-based with Acceptable Use Policy; Hugging Face tag license:other",
        paper="Falcon-Edge release blog, https://falcon-lm.github.io/blog/falcon-edge/",
        code_source=f"as falcon-e-1b (transformers BitLinear @ {TRANSFORMERS_BITNET_COMMIT}; onebitllms @ {ONEBITLLMS_COMMIT})",
        native_ternary_claim="as falcon-e-1b",
        scale_convention="divide",
        norm_names=("input_layernorm", "post_attention_layernorm"),
        tied_head=False,
        default_out="results/workload_profile/falcon-e-3b",
        act_quant_source="as falcon-e-1b",
        act_rounding="as falcon-e-1b",
        weight_rounding="as falcon-e-1b",
        prose_normalization="RMSNorm x2 per layer (input_layernorm, post_attention_layernorm) plus final model.norm; no sub-norms",
        prose_attention="Llama GQA, RoPE on q/k, no bias",
        prose_mlp="down_proj(silu(gate_proj(x)) * up_proj(x))  (SwiGLU)",
        latent_repo="tiiuae/Falcon-E-3B-Base",
        latent_revision="5c11e9b4141ad5155a11138874acd8124260efca",
        latent_note="'prequantized' branch; sharded, not used for cross-check in EXP-0002B.",
        crosscheck_tensors=(),
    ),
}


def get_spec(slug: str) -> ModelSpec:
    try:
        return SPECS[slug]
    except KeyError:
        raise SystemExit(f"unknown model '{slug}'; known: {', '.join(SPECS)}")
