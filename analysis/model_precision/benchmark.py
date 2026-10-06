"""
Memory and latency measurements for EXP-0004 configurations.

Latency numbers reflect eager PyTorch on Windows with per-forward ternary weight unpacking
(transformers BitLinear) and TORCHDYNAMO_DISABLE=1. Absolute values are not representative
of any optimized runtime; only the relative comparison across configurations is meaningful.
"""

from __future__ import annotations

import time

import torch

from common import cache_bytes, cuda_mem, make_cache, reset_peak, sync


@torch.inference_mode()
def kv_memory_sweep(model, tokenizer, token_ids: torch.Tensor, cache_kind: str, contexts: list[int], device) -> list[dict]:
    """Fill a cache with `ctx` prompt tokens (prefill, logits kept only for the last position), do one decode step,
    and report cache bytes and peak allocation. Also computes the theoretical BF16 bytes for comparison."""
    cfg = model.config
    L, nKV, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    rows = []
    for ctx in contexts:
        if ctx > cfg.max_position_embeddings or ctx + 1 > token_ids.numel():
            rows.append({"cache_kind": cache_kind, "ctx": ctx, "skipped": "beyond model max context or dataset"})
            continue
        ids = token_ids[:ctx].unsqueeze(0).to(device)
        cache = make_cache(model, cache_kind)
        reset_peak(); sync(); t0 = time.time()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        sync(); t_prefill = time.time() - t0
        peak_prefill = cuda_mem()
        nxt = out.logits[0, -1].argmax().view(1, 1)
        reset_peak(); sync(); t1 = time.time()
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        sync(); t_step = time.time() - t1
        cb = cache_bytes(cache)
        bf16_theory = 2 * nKV * hd * 2 * L * (ctx + 1)
        rows.append({"cache_kind": cache_kind, "ctx": ctx, "prefill_s": t_prefill, "one_decode_step_s": t_step,
                     "cache_total_bytes": cb["total_bytes"], "cache_residual_bytes": cb["residual_bytes"], "cache_quantized_bytes": cb["quantized_bytes"],
                     "bf16_theoretical_bytes": bf16_theory, "ratio_vs_bf16_theory": bf16_theory / cb["total_bytes"] if cb["total_bytes"] else None,
                     "effective_bits_per_kv_element": 8 * cb["total_bytes"] / (2 * nKV * hd * L * (ctx + 1)),
                     "peak_prefill_max_allocated_bytes": peak_prefill.get("max_allocated_bytes"),
                     "peak_decode_step_max_allocated_bytes": cuda_mem().get("max_allocated_bytes")})
        del cache, out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


@torch.inference_mode()
def latency(model, tokenizer, prompt_ids: torch.Tensor, cache_kind: str, new_tokens: int, device, warmup: int = 3, repeats: int = 5) -> dict:
    """Greedy decode latency with warmup and repeats; synchronized timers."""
    ids = prompt_ids.unsqueeze(0).to(device)

    def one():
        cache = make_cache(model, cache_kind)
        sync(); t0 = time.time()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        sync(); tp = time.time() - t0
        nxt = out.logits[0, -1].argmax()
        t1 = time.time()
        for _ in range(new_tokens - 1):
            out = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
            nxt = out.logits[0, -1].argmax()
        sync(); td = time.time() - t1
        return tp, td

    for _ in range(warmup):
        one()
    reset_peak()
    tps, tds = [], []
    for _ in range(repeats):
        tp, td = one(); tps.append(tp); tds.append(td)
    tps.sort(); tds.sort()
    return {"cache_kind": cache_kind, "prompt_tokens": int(ids.shape[1]), "new_tokens": new_tokens, "warmup": warmup, "repeats": repeats,
            "prefill_s_median": tps[len(tps) // 2], "prefill_s_min": tps[0], "ttft_s_median": tps[len(tps) // 2],
            "decode_s_median": tds[len(tds) // 2], "decode_tokens_per_s_median": (new_tokens - 1) / tds[len(tds) // 2],
            "latency_per_token_ms_median": 1e3 * tds[len(tds) // 2] / (new_tokens - 1), "peak_cuda": cuda_mem()}
