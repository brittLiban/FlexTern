"""Quick check: single-token decode speed with stock BitLinear (unpack per forward) vs cached unpacked weights,
and logit equivalence between the two. Writes results/model_precision/unpack_cache_check.json."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common import cache_unpacked_ternary_weights, cuda_mem, load_model, make_cache, sync, write_json  # noqa: E402


def bench(model, ids, steps=40):
    with torch.inference_mode():
        cache = make_cache(model, "dynamic")
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[0, -1].argmax()
        for _ in range(5):
            out = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True); nxt = out.logits[0, -1].argmax()
        sync(); t = time.time()
        for _ in range(steps):
            out = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True); nxt = out.logits[0, -1].argmax()
        sync(); dt = time.time() - t
    return {"tokens_per_s": steps / dt, "ms_per_token": 1e3 * dt / steps, "allocated_bytes": cuda_mem()["allocated_bytes"]}


def main():
    model, tok = load_model("cuda")
    ids = tok("The history of computing begins with", return_tensors="pt").input_ids.cuda()
    stock = bench(model, ids)
    print("stock BitLinear:", stock, flush=True)
    with torch.inference_mode():
        ref = model(input_ids=ids).logits[0, -1].float().clone()
    info = cache_unpacked_ternary_weights(model)
    with torch.inference_mode():
        new = model(input_ids=ids).logits[0, -1].float()
    diff = float((ref - new).abs().max())
    cached = bench(model, ids)
    print("cached unpack:", cached, "unpack info:", info, "max |logit diff|:", diff, flush=True)
    write_json("unpack_cache_check.json", {"stock": stock, "cached_unpacked": cached, "unpack_info": info, "max_abs_logit_diff": diff})


if __name__ == "__main__":
    main()
