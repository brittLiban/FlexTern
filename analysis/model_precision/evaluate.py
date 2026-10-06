"""
Quality evaluation for EXP-0004.

Two independent measurements:

1. lm-eval-harness tasks (0-shot loglikelihood): WikiText-2 word perplexity, HellaSwag, PIQA,
   ARC-Easy, ARC-Challenge, WinoGrande. These run ONE forward pass per candidate with no
   incremental KV cache, so they measure the HEAD configuration only; KV-cache configurations
   score identically to their head-equivalents by construction (verified once, see driver).

2. Teacher-forced incremental-decode perplexity through the REAL cache: for fixed WikiText-2
   windows, prefill P tokens, then step one token at a time (feeding the ground-truth token)
   with the chosen cache implementation, accumulating next-token NLL. This exercises prefill,
   lazy cache init, residual accumulation, re-quantization, retrieval and attention. The
   baseline control (DynamicCache) must reproduce the single-forward NLL on the same windows;
   quantized caches are compared against that incremental baseline.

Identical windows, tokenizer, seeds and settings are used for every configuration.
"""

from __future__ import annotations

import math
import time

import torch

from common import SEED, make_cache, seed_everything, sync, reset_peak, cuda_mem, cache_bytes

LM_EVAL_TASKS = ["wikitext", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"]


LM_EVAL_MAX_LENGTH = 2048   # rolling-perplexity window and request truncation; the model's 32k max would need 32k x 32k logits


def run_lm_eval(model, tokenizer, tasks=LM_EVAL_TASKS, limit=None, batch_size=4, max_length=LM_EVAL_MAX_LENGTH) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    seed_everything(SEED)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, dtype=torch.bfloat16, max_length=max_length)
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0, limit=limit, random_seed=SEED,
                                  numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED, log_samples=False)
    out = {"wall_s": time.time() - t0, "limit": limit, "max_length": max_length, "batch_size": batch_size, "tasks": {}}
    for task, r in res["results"].items():
        keep = {k: v for k, v in r.items() if any(k.startswith(p) for p in ("acc", "word_perplexity", "byte_perplexity", "bits_per_byte"))}
        keep["n_samples"] = res.get("n-samples", {}).get(task, {}).get("effective")
        out["tasks"][task] = keep
    out["versions"] = res.get("versions", {})
    out["config"] = {k: v for k, v in res.get("config", {}).items() if k in ("model", "model_args", "batch_size", "limit", "num_fewshot")}
    return out


def wikitext_windows(tokenizer, n_windows: int, ctx: int, seed: int = SEED) -> tuple[torch.Tensor, dict]:
    """Fixed, non-overlapping windows of ctx tokens from the WikiText-2 (raw) test split, deterministic."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total = ids.numel()
    max_windows = total // ctx
    n = min(n_windows, max_windows)
    starts = [i * ctx for i in range(n)]           # deterministic: first n windows, stride ctx
    windows = torch.stack([ids[s:s + ctx] for s in starts])
    meta = {"dataset": "Salesforce/wikitext wikitext-2-raw-v1 test", "total_tokens": int(total), "ctx": ctx, "n_windows": n,
            "window_starts": starts, "coverage_fraction": n * ctx / total}
    return windows, meta


@torch.inference_mode()
def single_forward_nll(model, windows: torch.Tensor, prefill: int, device) -> dict:
    """Reference: one forward per window; NLL of tokens at positions >= prefill."""
    nll_sum = 0.0; count = 0
    per_window = []
    for w in windows:
        ids = w.unsqueeze(0).to(device)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        targets = ids[0, 1:]
        nll = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        sel = nll[prefill - 1:]           # predictions for positions prefill..ctx-1
        per_window.append(float(sel.mean()))
        nll_sum += float(sel.sum()); count += sel.numel()
    return {"mean_nll": nll_sum / count, "ppl": math.exp(nll_sum / count), "tokens_scored": count, "per_window_nll": per_window}


@torch.inference_mode()
def incremental_nll(model, windows: torch.Tensor, prefill: int, cache_kind: str, device, bucket: int = 256,
                    progress=None) -> dict:
    """Prefill P tokens, then decode one token at a time with teacher forcing through the chosen cache."""
    ctx = windows.shape[1]
    nll_sum = 0.0; count = 0
    buckets: dict[int, list[float]] = {}
    per_window = []
    cache_stats_last = None
    t0 = time.time()
    reset_peak()
    for wi, w in enumerate(windows):
        ids = w.unsqueeze(0).to(device)
        cache = make_cache(model, cache_kind)
        out = model(input_ids=ids[:, :prefill], past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1].float()
        wsum = 0.0; wcount = 0
        for pos in range(prefill, ctx):
            target = ids[0, pos]
            nll = float(torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
            wsum += nll; wcount += 1
            buckets.setdefault(pos // bucket * bucket, []).append(nll)
            if pos == ctx - 1:
                break
            out = model(input_ids=target.view(1, 1), past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1].float()
        nll_sum += wsum; count += wcount
        per_window.append(wsum / wcount)
        cache_stats_last = cache_bytes(cache)
        if progress:
            progress(wi + 1, len(windows), wsum / wcount)
        del cache, out
    sync()
    return {"cache_kind": cache_kind, "prefill": prefill, "ctx": ctx, "mean_nll": nll_sum / count, "ppl": math.exp(nll_sum / count),
            "tokens_scored": count, "per_window_nll": per_window,
            "nll_by_position_bucket": {str(k): sum(v) / len(v) for k, v in sorted(buckets.items())},
            "last_window_cache_bytes": cache_stats_last, "peak_cuda": cuda_mem(), "wall_s": time.time() - t0,
            "decode_steps": count}


@torch.inference_mode()
def generate_sample(model, tokenizer, prompt_ids: torch.Tensor, cache_kind: str, max_new_tokens: int, device) -> dict:
    """Greedy generation with the chosen cache; returns text, timing, peak memory, cache bytes."""
    cache = make_cache(model, cache_kind)
    ids = prompt_ids.unsqueeze(0).to(device)
    reset_peak(); sync(); t0 = time.time()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    sync(); t_prefill = time.time() - t0
    next_tok = out.logits[0, -1].argmax()
    generated = [int(next_tok)]
    t1 = time.time()
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_tok.view(1, 1), past_key_values=cache, use_cache=True)
        next_tok = out.logits[0, -1].argmax()
        generated.append(int(next_tok))
        if int(next_tok) == tokenizer.eos_token_id:
            break
    sync(); t_decode = time.time() - t1
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"cache_kind": cache_kind, "prompt_tokens": int(ids.shape[1]), "new_tokens": len(generated), "text": text,
            "prefill_s": t_prefill, "decode_s": t_decode, "decode_tokens_per_s": (len(generated) - 1) / t_decode if t_decode > 0 and len(generated) > 1 else None,
            "ttft_s": t_prefill, "cache_bytes": cache_bytes(cache), "peak_cuda": cuda_mem()}
