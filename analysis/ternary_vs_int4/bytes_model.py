"""Bytes streamed per decoded token, from the actual GGUF tensor types (EXP-0011).

Decode on a CPU is largely memory-bound: each generated token reads every weight matrix once, plus the KV cache. File size is a poor proxy
(the embedding table is looked up, not streamed; the output head is streamed). This module reads the GGUF header with gguf-py (from the
llama.cpp checkout) and reports, per model:
  weight_bytes_streamed = sum of n_bytes of every tensor except the token-embedding lookup table, plus the token embedding table itself if it
                          doubles as the output head (no separate output.weight tensor);
  kv_bytes(depth)       = 2 (K and V) * n_layer * n_kv_head * head_dim * depth * 2 bytes (f16 cache, the llama-bench default).
It also decomposes bytes by tensor type so a high-precision head or embedding in a "ternary" model is visible.
"""
from __future__ import annotations

import sys
from collections import defaultdict

from . import config as C


def _reader(path):
    sys.path.insert(0, str(C.EXP_ROOT / "BitNet" / "3rdparty" / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415
    return GGUFReader(str(path), "r")


def _scalar(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    return int(f.parts[f.data[0]][0])


def tensor_bytes(t) -> int:
    """gguf-py's n_bytes is wrong for bitnet.cpp's I2_S type (2 bits/weight + a 32-byte per-tensor scale block); correct it explicitly."""
    if t.tensor_type.name == "I2_S":
        return int(t.n_elements) // 4 + 32
    return int(t.n_bytes)


def profile(model_id: str) -> dict:
    path = C.MODELS_DIR / C.MODELS[model_id]["file"]
    r = _reader(path)
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    n_layer = _scalar(r, f"{arch}.block_count")
    n_head = _scalar(r, f"{arch}.attention.head_count")
    n_kv = _scalar(r, f"{arch}.attention.head_count_kv") or n_head
    emb = _scalar(r, f"{arch}.embedding_length")
    head_dim = _scalar(r, f"{arch}.attention.key_length") or (emb // n_head)
    names = {t.name for t in r.tensors}
    has_out = "output.weight" in names
    by_type = defaultdict(int)
    streamed = 0
    total = 0
    for t in r.tensors:
        nb = tensor_bytes(t)
        total += nb
        if t.name == "token_embd.weight" and has_out:
            continue                                   # lookup only
        by_type[t.tensor_type.name] += nb
        streamed += nb
    return {"model": model_id, "arch": arch, "n_layer": n_layer, "n_head": n_head, "n_kv_head": n_kv, "head_dim": head_dim, "embedding_length": emb,
            "output_head": "separate output.weight" if has_out else "tied to token_embd.weight",
            "weight_bytes_streamed": streamed, "sum_all_tensor_bytes": total, "bytes_by_type": dict(by_type), "file_bytes": path.stat().st_size,
            "kv_bytes_per_token": 2 * n_layer * n_kv * head_dim * 2}


def decode_bytes(prof: dict, depth: float) -> float:
    return prof["weight_bytes_streamed"] + prof["kv_bytes_per_token"] * depth


def bytes_ratio(prof_int4: dict, prof_tern: dict, depth: float) -> float:
    """(bytes the INT4 model streams per token) / (bytes the ternary model streams) at a given KV depth: the decode speed-up if purely bandwidth-bound."""
    return decode_bytes(prof_int4, depth) / decode_bytes(prof_tern, depth)


if __name__ == "__main__":
    import json
    out = {m: profile(m) for m in C.MODELS if (C.MODELS_DIR / C.MODELS[m]["file"]).exists()}
    print(json.dumps(out, indent=1))
