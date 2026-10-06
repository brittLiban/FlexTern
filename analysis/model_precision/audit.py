"""
EXP-0004 feasibility audit: verify loading path, checkpoint identity, head execution, actual
KV dtype, and the internals of the quantized cache after a real generation. Writes
results/model_precision/audit.json and environment.json.

Usage (from repo root): python analysis/model_precision/audit.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (KV_INT4, cache_bytes, cuda_mem, environment, load_model, make_cache,  # noqa: E402
                    quantized_layer_internals, reset_peak, seed_everything, sync, verify_checkpoint, write_json)


def main() -> int:
    import torch
    seed_everything()
    env = environment()
    write_json("environment.json", env)
    print({k: env[k] for k in ("torch", "torch_cuda", "cuda_available", "transformers", "gpu") if k in env})

    ident = verify_checkpoint()
    print("checkpoint:", ident["sha256"][:16], "matches ADR-0004:", ident["matches_adr_0004"])

    t0 = time.time()
    reset_peak()
    model, tok = load_model("cuda")
    sync()
    audit = {"checkpoint": ident, "load_s": time.time() - t0, "memory_after_load": cuda_mem()}

    # Ternary layers stay packed uint8? Head is plain BF16 Linear? Embedding separate?
    q = model.model.layers[0].self_attn.q_proj
    audit["q_proj_class"] = type(q).__name__
    audit["q_proj_weight"] = {"dtype": str(q.weight.dtype), "shape": list(q.weight.shape)}
    audit["q_proj_weight_scale"] = {"dtype": str(q.weight_scale.dtype), "value": float(q.weight_scale.float().reshape(-1)[0])}
    head = model.lm_head
    audit["lm_head"] = {"class": type(head).__name__, "dtype": str(head.weight.dtype), "shape": list(head.weight.shape),
                        "bytes": head.weight.numel() * head.weight.element_size(),
                        "tied_to_embedding": head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()}
    audit["embed_tokens"] = {"dtype": str(model.model.embed_tokens.weight.dtype), "shape": list(model.model.embed_tokens.weight.shape)}
    audit["config"] = {k: getattr(model.config, k, None) for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                                                                   "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
                                                                   "tie_word_embeddings", "hidden_act", "model_type")}
    audit["quantization_config"] = str(getattr(model.config, "quantization_config", None))
    audit["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    audit["packed_ternary_bytes_resident"] = sum(m.weight.numel() * m.weight.element_size() for m in model.modules() if type(m).__name__ == "BitLinear")

    # Reference generation and actual KV dtype through the default cache path
    prompt = "The Technology Innovation Institute is"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        cache = make_cache(model, "dynamic")
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        layer0 = cache.layers[0]
        audit["default_cache"] = {"cache_class": type(cache).__name__, "layer_class": type(layer0).__name__,
                                  "keys_dtype": str(layer0.keys.dtype), "keys_shape": list(layer0.keys.shape),
                                  "values_dtype": str(layer0.values.dtype), "values_shape": list(layer0.values.shape),
                                  "bytes_after_prefill": cache_bytes(cache)}
        gen = model.generate(ids, max_new_tokens=24, do_sample=False)
        audit["greedy_sample"] = tok.decode(gen[0], skip_special_tokens=True)

        # Quantized cache internals after a generation long enough to flush the residual at least twice
        q_ids = tok("Write a short paragraph about the history of computing. " * 8, return_tensors="pt").input_ids.cuda()
        qcache = make_cache(model, "int4")
        out = model(input_ids=q_ids, past_key_values=qcache, use_cache=True)
        nxt = out.logits[0, -1].argmax()
        for _ in range(300):
            out = model(input_ids=nxt.view(1, 1), past_key_values=qcache, use_cache=True)
            nxt = out.logits[0, -1].argmax()
        audit["quantized_cache_settings"] = KV_INT4
        audit["quantized_cache_layer0_internals"] = quantized_layer_internals(qcache, 0)
        audit["quantized_cache_bytes"] = cache_bytes(qcache)
        total_tokens = q_ids.shape[1] + 301
        cfg = model.config
        bf16_equiv = 2 * cfg.num_key_value_heads * cfg.head_dim * 2 * cfg.num_hidden_layers * total_tokens
        audit["quantized_cache_tokens"] = total_tokens
        audit["quantized_cache_bf16_equivalent_bytes"] = bf16_equiv
        audit["quantized_cache_effective_bits_per_element_incl_residual"] = 8 * cache_bytes(qcache)["total_bytes"] / (2 * cfg.num_key_value_heads * cfg.head_dim * cfg.num_hidden_layers * total_tokens)
    audit["peak_memory_during_audit"] = cuda_mem()
    write_json("audit.json", audit)
    print("lm_head:", audit["lm_head"]); print("q_proj:", audit["q_proj_weight"]); print("default cache:", audit["default_cache"]["keys_dtype"], audit["default_cache"]["keys_shape"])
    print("quantized layer0:", audit["quantized_cache_layer0_internals"])
    print("sample:", audit["greedy_sample"][:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
