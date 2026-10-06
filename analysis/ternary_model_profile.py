"""
FlexTern workload characterization of a real ternary transformer (EXP-0002).

Analyzes microsoft/bitnet-b1.58-2B-4T (packed 2-bit checkpoint) and writes
machine-readable CSV/JSON plus a human-readable summary under
results/workload_profile/. Everything is derived from the checkpoint header,
the packed weight bytes, config.json, and the reference implementation's
quantization code. Nothing is executed at runtime; all arithmetic counts are
static and labeled INFERRED.

Usage (from repo root):
    python analysis/ternary_model_profile.py
    python analysis/ternary_model_profile.py --local-file path/to/model.safetensors --local-config path/to/config.json

Deterministic. No random numbers are used anywhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import SafetensorsFile, unpack_ternary  # noqa: E402

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
MODEL_REVISION = "04c3b9ad9361b824064a1f25ea60a8be9599b127"
EXPECTED_SHA256 = "8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b"
EXPECTED_FILE_SIZE = 1_178_623_988
TRANSFORMERS_BITNET_COMMIT = "d6abd1333078195535f461589b64f112b53688b2"  # integrations/bitnet.py

# Activation quantization facts from transformers/integrations/bitnet.py
# (BitLinear.activation_quant and ActQuant.forward): per-token absmax,
# scale = 127 / absmax, round, clamp to [-128, 127]. Because |x * scale| <= 127
# before rounding, the value -128 is unreachable; the effective range is
# [-127, +127].
ACT_BITS = 8
ACT_FORMAT_MIN, ACT_FORMAT_MAX = -128, 127
ACT_MODEL_MIN, ACT_MODEL_MAX = -127, 127

PACKED_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
SCALE_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight_scale$")
NORM_RE = re.compile(r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm|self_attn\.attn_sub_norm|mlp\.ffn_sub_norm)\.weight$")
FINAL_NORM = "model.norm.weight"
EMBED = "model.embed_tokens.weight"


# =============================================================================
# Data records
# =============================================================================
@dataclass
class TernaryTensorStats:
    name: str
    layer: int
    module: str           # q_proj ... down_proj
    group: str            # attention | mlp
    packed_shape: str
    out_dim: int          # N
    in_dim: int           # K (dot-product length)
    numel: int
    count_neg: int
    count_zero: int
    count_pos: int
    count_code3: int
    pct_neg: float
    pct_zero: float
    pct_pos: float
    weight_scale: float
    row_zero_rate_min: float
    row_zero_rate_max: float
    row_zero_rate_std: float
    max_nonzero_per_row: int
    min_nonzero_per_row: int
    mean_nonzero_per_row: float
    allzero_rows: int          # output channels whose entire weight row is 0
    allzero_cols: int          # input positions whose entire weight column is 0
    g16_allzero_rate_live_rows: float  # g=16 all-zero rate excluding all-zero rows
    g4_allzero_rate: float
    g4_predicted: float
    g8_allzero_rate: float
    g8_predicted: float
    g16_allzero_rate: float
    g16_predicted: float
    packed_bytes: int


# =============================================================================
# Helpers
# =============================================================================
def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def signed_bits_for_range(lo: int, hi: int) -> int:
    """Smallest b such that a signed b-bit integer holds every value in [lo, hi]."""
    b = 1
    while not (-(1 << (b - 1)) <= lo and hi <= (1 << (b - 1)) - 1):
        b += 1
    return b


def accumulator_width(k: int, act_min: int, act_max: int) -> dict:
    """Exact extreme sums of K products a_i * w_i, w_i in {-1,0,+1}, a_i in [act_min, act_max].

    Each term's extremes: max term = max(act_max, -act_min); min term = -max term.
    (With w = -1, the most negative activation becomes the most positive product.)
    """
    term_max = max(act_max, -act_min)
    hi = k * term_max
    lo = -hi
    return {"K": k, "act_min": act_min, "act_max": act_max, "sum_min": lo, "sum_max": hi,
            "signed_bits": signed_bits_for_range(lo, hi)}


def brute_force_accumulator(k: int, act_min: int, act_max: int) -> tuple[int, int]:
    """Enumerate every activation/weight combination for tiny K. Validation only."""
    import itertools
    acts = range(act_min, act_max + 1)
    ws = (-1, 0, 1)
    lo, hi = None, None
    for a in itertools.product(acts, repeat=k):
        for w in itertools.product(ws, repeat=k):
            s = sum(x * y for x, y in zip(a, w))
            lo = s if lo is None or s < lo else lo
            hi = s if hi is None or s > hi else hi
    return lo, hi


def analyze_ternary_tensor(st: SafetensorsFile, name: str, scale_name: str) -> TernaryTensorStats:
    m = PACKED_RE.match(name)
    layer, grp, mod = int(m.group(1)), m.group(2), m.group(3)
    packed = st.uint8(name)
    q = unpack_ternary(packed)               # int8 [N, K]
    n, k = q.shape
    neg = int(np.count_nonzero(q == -1))
    zero = int(np.count_nonzero(q == 0))
    pos = int(np.count_nonzero(q == 1))
    code3 = int(np.count_nonzero(q == 2))
    numel = int(q.size)
    is_zero = (q == 0)
    row_zero = is_zero.mean(axis=1)
    nz_per_row = k - is_zero.sum(axis=1)
    p = zero / numel
    dead_rows = int((nz_per_row == 0).sum())
    dead_cols = int(is_zero.all(axis=0).sum())
    live = is_zero[nz_per_row > 0]
    g16_live = float(live.reshape(live.shape[0], k // 16, 16).all(axis=2).mean()) if k % 16 == 0 and live.shape[0] else float("nan")
    out = dict(
        name=name, layer=layer, module=mod, group="attention" if grp == "self_attn" else "mlp",
        packed_shape="x".join(str(s) for s in packed.shape), out_dim=n, in_dim=k, numel=numel,
        count_neg=neg, count_zero=zero, count_pos=pos, count_code3=code3,
        pct_neg=100.0 * neg / numel, pct_zero=100.0 * zero / numel, pct_pos=100.0 * pos / numel,
        weight_scale=float(st.bf16_as_f32(scale_name).reshape(-1)[0]),
        row_zero_rate_min=float(row_zero.min()), row_zero_rate_max=float(row_zero.max()),
        row_zero_rate_std=float(row_zero.std()),
        max_nonzero_per_row=int(nz_per_row.max()), min_nonzero_per_row=int(nz_per_row.min()),
        mean_nonzero_per_row=float(nz_per_row.mean()),
        allzero_rows=dead_rows, allzero_cols=dead_cols, g16_allzero_rate_live_rows=g16_live,
        packed_bytes=int(packed.nbytes),
    )
    for g in (4, 8, 16):
        if k % g == 0:
            rate = float(is_zero.reshape(n, k // g, g).all(axis=2).mean())
        else:
            rate = float("nan")
        out[f"g{g}_allzero_rate"] = rate
        out[f"g{g}_predicted"] = p ** g
    return TernaryTensorStats(**out)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt_bytes(b: float) -> str:
    return f"{b / 1e6:,.1f} MB"


# =============================================================================
# Main analysis
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-file", help="Path to model.safetensors (skips download)")
    ap.add_argument("--local-config", help="Path to config.json (skips download)")
    ap.add_argument("--out", default=os.path.join("results", "workload_profile"))
    ap.add_argument("--context", type=int, default=4096, help="context length for attention/KV estimates")
    ap.add_argument("--skip-hash", action="store_true", help="skip SHA-256 of the 1.18 GB file")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    argv = " ".join(sys.argv)

    # ---- 1. Resolve files ------------------------------------------------------
    if args.local_file and args.local_config:
        model_path, config_path = args.local_file, args.local_config
        source = f"local files ({model_path})"
    else:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
        config_path = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
        source = f"huggingface_hub: {MODEL_ID} @ {MODEL_REVISION}"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    file_sha = None if args.skip_hash else sha256_file(model_path)
    size_ok = os.path.getsize(model_path) == EXPECTED_FILE_SIZE
    hash_ok = (file_sha == EXPECTED_SHA256) if file_sha else None

    # ---- 2. Config -------------------------------------------------------------
    L = cfg["num_hidden_layers"]; H = cfg["hidden_size"]; I = cfg["intermediate_size"]
    nH = cfg["num_attention_heads"]; nKV = cfg["num_key_value_heads"]; V = cfg["vocab_size"]
    hd = H // nH
    ctx = args.context
    model_summary = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "source": source,
        "license": "MIT (LICENSE file in model repo; Hugging Face tag license:mit)",
        "paper": "BitNet b1.58 2B4T Technical Report, arXiv:2504.12285",
        "architectures": cfg["architectures"], "model_type": cfg["model_type"],
        "num_hidden_layers": L, "hidden_size": H, "intermediate_size": I,
        "num_attention_heads": nH, "num_key_value_heads": nKV, "head_dim": hd,
        "gqa_group_size": nH // nKV, "vocab_size": V, "embedding_dim": H,
        "max_position_embeddings": cfg["max_position_embeddings"],
        "rope_theta": cfg["rope_theta"], "rms_norm_eps": cfg["rms_norm_eps"],
        "hidden_act": cfg["hidden_act"], "tie_word_embeddings": cfg["tie_word_embeddings"],
        "torch_dtype": cfg["torch_dtype"], "quantization_config": cfg["quantization_config"],
        "normalization": "RMSNorm (BitNetRMSNorm) x4 per layer: input_layernorm, post_attention_layernorm, "
                         "self_attn.attn_sub_norm (before o_proj), mlp.ffn_sub_norm (before down_proj); plus final model.norm",
        "attention": "GQA, RoPE on q/k, attn_sub_norm applied to attention output before o_proj, no bias",
        "mlp": "down_proj(ffn_sub_norm(relu2(gate_proj(x)) * up_proj(x)))",
        "weight_quantization": "absmean ternary (WeightQuant: scale=1/mean|W|, round, clamp[-1,1]); stored packed 2-bit "
                               "(code = q+1) 4 per uint8 along dim 0; one BF16 weight_scale per tensor; "
                               "AutoBitLinear offline mode: output = linear(x, W_q) * weight_scale",
        "activation_quantization": "per-token (last dim) absmax symmetric int8: scale = 127/absmax, round, clamp[-128,127]; "
                                   "effective range [-127,127]; dequantized by dividing by the same scale",
        "config_source": "config.json at the pinned revision",
        "code_source": f"huggingface/transformers src/transformers/integrations/bitnet.py @ {TRANSFORMERS_BITNET_COMMIT}",
    }

    # ---- 3. Open checkpoint, classify tensors, reconcile bytes ----------------
    st = SafetensorsFile(model_path)
    layout = st.verify_layout()
    classes = defaultdict(list)
    for name, t in st.tensors.items():
        if PACKED_RE.match(name) and t.dtype == "U8":
            classes["packed_ternary"].append(name)
        elif SCALE_RE.match(name):
            classes["weight_scale"].append(name)
        elif NORM_RE.match(name) or name == FINAL_NORM:
            classes["norm"].append(name)
        elif name == EMBED:
            classes["embedding"].append(name)
        else:
            classes["unclassified"].append(name)
    assert len(classes["unclassified"]) == 0, f"unclassified tensors: {classes['unclassified']}"
    assert len(classes["packed_ternary"]) == 7 * L
    assert len(classes["weight_scale"]) == 7 * L
    assert len(classes["norm"]) == 4 * L + 1
    assert len(classes["embedding"]) == 1
    assert "lm_head.weight" not in st.tensors, "expected tied head (no lm_head tensor)"
    assert layout["tensor_count"] == 7 * L + 7 * L + 4 * L + 1 + 1
    assert st.tensors[EMBED].shape == (V, H)

    # ---- 4. Ternary tensor statistics -----------------------------------------
    expected_out = {"q_proj": nH * hd, "k_proj": nKV * hd, "v_proj": nKV * hd, "o_proj": H,
                    "gate_proj": I, "up_proj": I, "down_proj": H}
    expected_in = {"q_proj": H, "k_proj": H, "v_proj": H, "o_proj": nH * hd,
                   "gate_proj": H, "up_proj": H, "down_proj": I}
    stats: list[TernaryTensorStats] = []
    for name in sorted(classes["packed_ternary"], key=lambda s: (int(PACKED_RE.match(s).group(1)), s)):
        s = analyze_ternary_tensor(st, name, name + "_scale")
        assert s.out_dim == expected_out[s.module], f"{name}: out {s.out_dim} != {expected_out[s.module]}"
        assert s.in_dim == expected_in[s.module], f"{name}: in {s.in_dim} != {expected_in[s.module]}"
        assert s.count_neg + s.count_zero + s.count_pos + s.count_code3 == s.numel
        assert s.count_code3 == 0, f"{name}: reserved 2-bit code 3 present"
        stats.append(s)
        print(f"  {name:48s} N={s.out_dim:5d} K={s.in_dim:4d} zero={s.pct_zero:5.2f}% scale={s.weight_scale:.5f}", flush=True)

    total_ternary = sum(s.numel for s in stats)
    tot_neg = sum(s.count_neg for s in stats); tot_zero = sum(s.count_zero for s in stats); tot_pos = sum(s.count_pos for s in stats)
    assert tot_neg + tot_zero + tot_pos == total_ternary
    expected_ternary = L * (H * (nH * hd) + 2 * H * (nKV * hd) + (nH * hd) * H + 2 * H * I + I * H)
    assert total_ternary == expected_ternary, (total_ternary, expected_ternary)
    packed_bytes_total = sum(s.packed_bytes for s in stats)

    zr = np.array([s.pct_zero for s in stats])
    global_dist = {
        "total_ternary_weights": total_ternary, "count_neg": tot_neg, "count_zero": tot_zero, "count_pos": tot_pos,
        "pct_neg": 100 * tot_neg / total_ternary, "pct_zero": 100 * tot_zero / total_ternary, "pct_pos": 100 * tot_pos / total_ternary,
        "pct_sum": 100 * (tot_neg + tot_zero + tot_pos) / total_ternary,
        "tensor_zero_rate_min_pct": float(zr.min()), "tensor_zero_rate_max_pct": float(zr.max()),
        "tensor_zero_rate_median_pct": float(np.median(zr)), "tensor_zero_rate_mean_unweighted_pct": float(zr.mean()),
        "tensor_zero_rate_size_weighted_mean_pct": 100 * tot_zero / total_ternary, "tensor_zero_rate_std_pct": float(zr.std()),
        "most_sparse_tensor": max(stats, key=lambda s: s.pct_zero).name,
        "least_sparse_tensor": min(stats, key=lambda s: s.pct_zero).name,
        "weight_scale_min": min(s.weight_scale for s in stats), "weight_scale_max": max(s.weight_scale for s in stats),
        "reserved_code3_count": sum(s.count_code3 for s in stats),
    }

    # by module type
    by_module = []
    for mod in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        ss = [s for s in stats if s.module == mod]
        n = sum(s.numel for s in ss); z = sum(s.count_zero for s in ss)
        ng = sum(s.count_neg for s in ss); ps = sum(s.count_pos for s in ss)
        by_module.append({"module": mod, "group": ss[0].group, "tensors": len(ss), "K": ss[0].in_dim, "N": ss[0].out_dim,
                          "weights": n, "pct_of_ternary": 100 * n / total_ternary,
                          "pct_zero": 100 * z / n, "pct_neg": 100 * ng / n, "pct_pos": 100 * ps / n,
                          "zero_rate_min_pct": min(s.pct_zero for s in ss), "zero_rate_max_pct": max(s.pct_zero for s in ss),
                          "scale_min": min(s.weight_scale for s in ss), "scale_max": max(s.weight_scale for s in ss)})
    by_layer = []
    for layer in range(L):
        ss = [s for s in stats if s.layer == layer]
        n = sum(s.numel for s in ss); z = sum(s.count_zero for s in ss)
        by_layer.append({"layer": layer, "weights": n, "pct_zero": 100 * z / n,
                         "attn_pct_zero": 100 * sum(s.count_zero for s in ss if s.group == "attention") / sum(s.numel for s in ss if s.group == "attention"),
                         "mlp_pct_zero": 100 * sum(s.count_zero for s in ss if s.group == "mlp") / sum(s.numel for s in ss if s.group == "mlp")})

    # ---- 4b. Dead channels: all-zero rows/cols and MLP intermediate-channel death
    dead_rows = []
    for layer in range(L):
        ss = {s.module: s for s in stats if s.layer == layer}
        # Recompute the row masks for gate/up to intersect them (cheap: two tensors per layer).
        g_dead = unpack_ternary(st.uint8(ss["gate_proj"].name)).any(axis=1) == False
        u_dead = unpack_ternary(st.uint8(ss["up_proj"].name)).any(axis=1) == False
        d_cols = unpack_ternary(st.uint8(ss["down_proj"].name)).any(axis=0) == False
        dead_rows.append({"layer": layer,
                          "q_dead_rows": ss["q_proj"].allzero_rows, "k_dead_rows": ss["k_proj"].allzero_rows,
                          "v_dead_rows": ss["v_proj"].allzero_rows, "o_dead_rows": ss["o_proj"].allzero_rows,
                          "gate_dead_rows": int(g_dead.sum()), "up_dead_rows": int(u_dead.sum()),
                          "gate_or_up_dead": int((g_dead | u_dead).sum()), "gate_and_up_dead": int((g_dead & u_dead).sum()),
                          "down_dead_cols": int(d_cols.sum()),
                          "down_cols_dead_where_gate_or_up_dead": int((d_cols & (g_dead | u_dead)).sum()),
                          "intermediate_size": I,
                          "pct_intermediate_channels_dead": 100 * float((g_dead | u_dead).sum()) / I})
    dead_summary = {
        "total_allzero_rows_all_tensors": sum(s.allzero_rows for s in stats),
        "total_allzero_cols_all_tensors": sum(s.allzero_cols for s in stats),
        "mlp_intermediate_channels_dead_total": sum(r["gate_or_up_dead"] for r in dead_rows),
        "mlp_intermediate_channels_total": I * L,
        "pct_mlp_intermediate_channels_dead": 100 * sum(r["gate_or_up_dead"] for r in dead_rows) / (I * L),
        "weights_in_dead_rows_or_cols": sum(s.allzero_rows * s.in_dim for s in stats),
        "pct_ternary_weights_in_dead_rows": 100 * sum(s.allzero_rows * s.in_dim for s in stats) / total_ternary,
    }
    print(f"  dead MLP intermediate channels: {dead_summary['mlp_intermediate_channels_dead_total']:,} of {I*L:,} ({dead_summary['pct_mlp_intermediate_channels_dead']:.2f}%)")

    # ---- 5. Dot-length distribution -------------------------------------------
    by_k = defaultdict(lambda: {"K": 0, "modules": set(), "tensors": 0, "weights": 0, "output_rows": 0, "zero": 0})
    for s in stats:
        d = by_k[s.in_dim]; d["K"] = s.in_dim; d["modules"].add(s.module); d["tensors"] += 1
        d["weights"] += s.numel; d["output_rows"] += s.out_dim; d["zero"] += s.count_zero
    dot_rows = []
    for k in sorted(by_k):
        d = by_k[k]
        dot_rows.append({"K": k, "modules": "+".join(sorted(d["modules"])), "tensors": d["tensors"],
                         "dot_products_per_token": d["output_rows"], "weights": d["weights"],
                         "pct_of_ternary_weights": 100 * d["weights"] / total_ternary, "pct_zero": 100 * d["zero"] / d["weights"],
                         "expected_nonzero_terms": k * (1 - d["zero"] / d["weights"])})
    # expected terms per representative dot product, per module
    expected_terms = []
    for m in by_module:
        k = m["K"]; p0 = m["pct_zero"] / 100; pn = m["pct_neg"] / 100; pp = m["pct_pos"] / 100
        expected_terms.append({"module": m["module"], "K": k, "expected_nonzero": k * (1 - p0),
                               "expected_pos": k * pp, "expected_neg": k * pn, "expected_zero": k * p0})

    # ---- 6. Operation inventory per transformer block (INFERRED, static) ------
    ops = []
    def add(name, cls, domain, count_per_block, macs_per_token, notes, K="", N=""):
        ops.append({"operation": name, "class": cls, "domain": domain, "count_per_block": count_per_block,
                    "K": K, "N": N, "macs_or_ops_per_token_per_block": macs_per_token, "notes": notes})
    for m in by_module:
        add(m["module"], "linear", "TERNARY", 1, m["weights"] // L, "int8 act x ternary w, one BF16 scale per tensor", m["K"], m["N"])
    add("activation_quant (per-token absmax int8)", "quantize", "NON-TERNARY", 4, 2 * H + 2 * (nH * hd) + 2 * H + 2 * I,
        "4 distinct inputs: pre-qkv (H), pre-o (H), pre-gate/up (H), pre-down (I); absmax+scale+round each", "", "")
    add("weight_scale multiply", "scale", "NON-TERNARY", 7, sum(m["N"] for m in by_module),
        "one multiply per output element per linear (BF16)", "", "")
    add("RMSNorm (input, post_attn, attn_sub, ffn_sub)", "norm", "NON-TERNARY", 4, 3 * H + I,
        "elements normalized: input_layernorm H, post_attention_layernorm H, attn_sub_norm H, ffn_sub_norm I; each ~3 ops/element", "", "")
    add("RoPE on q and k", "rotary", "NON-TERNARY", 1, 2 * (nH * hd + nKV * hd), "complex rotate, ~2 mul+add per element", "", "")
    add(f"attention QK^T @ context {ctx}", "attention", "NON-TERNARY", 1, nH * hd * ctx, "BF16/FP, grows linearly with context", hd, ctx)
    add(f"softmax @ context {ctx}", "softmax", "NON-TERNARY", 1, nH * ctx, "exp + sum + divide per score", "", "")
    add(f"attention AV @ context {ctx}", "attention", "NON-TERNARY", 1, nH * hd * ctx, "BF16/FP, grows linearly with context", ctx, hd)
    add("KV-cache write (BF16)", "memory", "NON-TERNARY", 1, 2 * nKV * hd, f"{2 * nKV * hd * 2} bytes/token/layer written", "", "")
    add(f"KV-cache read @ context {ctx} (BF16)", "memory", "NON-TERNARY", 1, 2 * nKV * hd * ctx, f"{2 * nKV * hd * ctx * 2} bytes/token/layer read", "", "")
    add("relu^2", "activation", "NON-TERNARY", 1, I, "max(0,x)^2 on gate output", "", "")
    add("gate*up elementwise", "elementwise", "NON-TERNARY", 1, I, "BF16 multiply", "", "")
    add("residual add", "elementwise", "NON-TERNARY", 2, 2 * H, "BF16 add", "", "")
    ternary_macs_per_block = sum(m["weights"] for m in by_module) // L
    attn_macs_per_block = 2 * nH * hd * ctx
    lm_head_macs = V * H
    op_summary = {
        "ternary_macs_per_token_per_block": ternary_macs_per_block,
        "ternary_macs_per_token_total": ternary_macs_per_block * L,
        "attention_score_macs_per_token_per_block_at_context": attn_macs_per_block,
        "attention_score_macs_per_token_total_at_context": attn_macs_per_block * L,
        "lm_head_bf16_macs_per_token": lm_head_macs,
        "context": ctx,
        "pct_linear_macs_ternary": 100 * ternary_macs_per_block * L / (ternary_macs_per_block * L + lm_head_macs),
        "pct_linear_macs_lm_head_bf16": 100 * lm_head_macs / (ternary_macs_per_block * L + lm_head_macs),
    }

    # ---- 7. Parameter accounting ----------------------------------------------
    embed_params = V * H
    norm_params = sum(st.tensors[n].numel for n in classes["norm"])
    scale_params = len(classes["weight_scale"])
    params = {
        "ternary_linear": total_ternary, "embedding_bf16_tied_lm_head": embed_params,
        "norm_bf16": norm_params, "weight_scales_bf16": scale_params,
        "total": total_ternary + embed_params + norm_params + scale_params,
    }
    params["pct_ternary"] = 100 * total_ternary / params["total"]
    params["pct_embedding"] = 100 * embed_params / params["total"]
    attn_w = sum(m["weights"] for m in by_module if m["group"] == "attention")
    mlp_w = sum(m["weights"] for m in by_module if m["group"] == "mlp")
    params["attention_proj_ternary"] = attn_w; params["mlp_ternary"] = mlp_w
    params["pct_of_ternary_attention"] = 100 * attn_w / total_ternary; params["pct_of_ternary_mlp"] = 100 * mlp_w / total_ternary

    # ---- 8. Memory representation ---------------------------------------------
    W = total_ternary
    bits_log2_3 = math.log2(3)
    mem_rows = [
        {"representation": "BF16 (16 bit/weight)", "scope": "ternary linear weights only", "bytes": 2 * W, "note": "THEORETICAL reference"},
        {"representation": "INT8 (8 bit/weight)", "scope": "ternary linear weights only", "bytes": W, "note": "THEORETICAL"},
        {"representation": "INT4 (4 bit/weight)", "scope": "ternary linear weights only", "bytes": W / 2, "note": "THEORETICAL"},
        {"representation": "2-bit naive ternary", "scope": "ternary linear weights only", "bytes": W / 4, "note": "THEORETICAL; equals checkpoint packing"},
        {"representation": "log2(3) = 1.585 bit/weight bound", "scope": "ternary linear weights only", "bytes": W * bits_log2_3 / 8, "note": "THEORETICAL information bound"},
        {"representation": "5 trits per byte (1.6 bit/weight)", "scope": "ternary linear weights only", "bytes": W / 5, "note": "THEORETICAL practical packing (3^5=243<=256)"},
        {"representation": "ACTUAL checkpoint packed U8", "scope": "ternary linear weights only", "bytes": packed_bytes_total, "note": "MEASURED from header byte ranges"},
        {"representation": "weight_scale BF16", "scope": "metadata", "bytes": 2 * scale_params, "note": "MEASURED; 1 per tensor"},
        {"representation": "norm weights BF16", "scope": "metadata", "bytes": 2 * norm_params, "note": "MEASURED"},
        {"representation": "embedding / tied lm_head BF16", "scope": "non-ternary weights", "bytes": st.tensors[EMBED].nbytes, "note": "MEASURED; 55.7% of file"},
        {"representation": "ACTUAL whole file", "scope": "model.safetensors", "bytes": layout["file_size"], "note": "MEASURED"},
        {"representation": "HYPOTHETICAL: 2-bit ternary layers + INT8 head", "scope": "all weights", "bytes": W / 4 + embed_params + 2 * (norm_params + scale_params), "note": "HYPOTHETICAL FlexTern packing"},
        {"representation": "HYPOTHETICAL: 2-bit ternary layers + INT4 head", "scope": "all weights", "bytes": W / 4 + embed_params / 2 + 2 * (norm_params + scale_params), "note": "HYPOTHETICAL; head quality impact unknown"},
    ]

    # ---- 9. Decode traffic model (INFERRED / ANALYTICAL) -----------------------
    layer_packed = packed_bytes_total
    head_bf16 = st.tensors[EMBED].nbytes
    meta = 2 * (norm_params + scale_params)
    kv_read = 2 * nKV * hd * 2 * L  # bytes per token per unit of context
    traffic = []
    def trow(name, layers_b, head_b, note):
        total = layers_b + head_b + meta
        traffic.append({"scenario": name, "ternary_layer_bytes_per_token": layers_b, "head_bytes_per_token": head_b,
                        "norm_and_scale_bytes_per_token": meta, "weight_bytes_per_token": total,
                        "pct_head": 100 * head_b / total, "kv_cache_read_bytes_per_token_at_context": kv_read * ctx,
                        "total_bytes_per_token_at_context": total + kv_read * ctx, "note": note})
    trow("As stored: 2-bit packed layers + BF16 head", layer_packed, head_bf16, "matches checkpoint")
    trow("BF16 layers + BF16 head (dense reference)", 2 * W, head_bf16, "what a BF16 model of this shape would stream")
    trow("INT8 layers + BF16 head", W, head_bf16, "")
    trow("INT4 layers + BF16 head", W / 2, head_bf16, "")
    trow("2-bit layers + INT8 head", layer_packed, embed_params, "HYPOTHETICAL head quantization")
    trow("2-bit layers + INT4 head", layer_packed, embed_params / 2, "HYPOTHETICAL head quantization")
    trow("1.6-bit (5 trits/byte) layers + INT4 head", W / 5, embed_params / 2, "HYPOTHETICAL")
    trow("2-bit layers only (head cached on-chip / excluded)", layer_packed, 0, "if the head were resident or handled elsewhere")

    # ---- 10. Accumulator widths -----------------------------------------------
    acc_rows = []
    max_nz = {k: max(s.max_nonzero_per_row for s in stats if s.in_dim == k) for k in by_k}
    for k in sorted(by_k):
        f = accumulator_width(k, ACT_FORMAT_MIN, ACT_FORMAT_MAX)
        a = accumulator_width(k, ACT_MODEL_MIN, ACT_MODEL_MAX)
        d = accumulator_width(max_nz[k], ACT_MODEL_MIN, ACT_MODEL_MAX)
        acc_rows.append({"K": k, "modules": "+".join(sorted(by_k[k]["modules"])),
                         "format_range": f"[{ACT_FORMAT_MIN},{ACT_FORMAT_MAX}]", "format_sum_min": f["sum_min"], "format_sum_max": f["sum_max"], "format_signed_bits": f["signed_bits"],
                         "model_range": f"[{ACT_MODEL_MIN},{ACT_MODEL_MAX}]", "model_sum_min": a["sum_min"], "model_sum_max": a["sum_max"], "model_signed_bits": a["signed_bits"],
                         "max_nonzero_in_any_row": max_nz[k], "data_bounded_sum_max": d["sum_max"], "data_bounded_signed_bits": d["signed_bits"],
                         "note": "format: INT8 [-128,127] with ternary -1 giving +128; model: activation_quant cannot produce -128; data-bounded: INFERRED from this checkpoint only"})
    # Also the Phase-1 reference point and some sub-block sizes a tiler might use
    for k in (16, 64, 256, 512, 1024):
        f = accumulator_width(k, ACT_FORMAT_MIN, ACT_FORMAT_MAX); a = accumulator_width(k, ACT_MODEL_MIN, ACT_MODEL_MAX)
        acc_rows.append({"K": k, "modules": "(tile size, not a model K)", "format_range": f"[{ACT_FORMAT_MIN},{ACT_FORMAT_MAX}]",
                         "format_sum_min": f["sum_min"], "format_sum_max": f["sum_max"], "format_signed_bits": f["signed_bits"],
                         "model_range": f"[{ACT_MODEL_MIN},{ACT_MODEL_MAX}]", "model_sum_min": a["sum_min"], "model_sum_max": a["sum_max"], "model_signed_bits": a["signed_bits"],
                         "max_nonzero_in_any_row": "", "data_bounded_sum_max": "", "data_bounded_signed_bits": "", "note": "partial-sum tile"})

    # ---- 11. Write outputs ----------------------------------------------------
    out = args.out
    model_summary.update({
        "file_size_bytes": layout["file_size"], "file_size_matches_expected": size_ok,
        "file_sha256": file_sha, "file_sha256_matches_hf_etag": hash_ok,
        "header_len": layout["header_len"], "tensor_count": layout["tensor_count"],
        "tensor_classes": {k: len(v) for k, v in classes.items()},
        "parameters": params, "global_ternary_distribution": global_dist, "operation_summary": op_summary,
        "dead_channels": dead_summary,
    })
    write_csv(os.path.join(out, "dead_channels_by_layer.csv"), dead_rows)
    with open(os.path.join(out, "model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)
    write_csv(os.path.join(out, "weight_distribution.csv"), [asdict(s) for s in stats])
    write_csv(os.path.join(out, "weight_distribution_by_module.csv"), by_module)
    write_csv(os.path.join(out, "weight_distribution_by_layer.csv"), by_layer)
    write_csv(os.path.join(out, "layer_shapes.csv"), [
        {"layer": s.layer, "module": s.module, "group": s.group, "op": "linear", "K_in": s.in_dim, "N_out": s.out_dim,
         "weight_shape_unpacked": f"{s.out_dim}x{s.in_dim}", "weight_shape_packed_u8": s.packed_shape, "weights": s.numel,
         "ternary": True, "has_scale": True, "scale_granularity": "per-tensor", "dot_length": s.in_dim,
         "dots_per_token": s.out_dim, "repeats_per_block": 1} for s in stats] + [
        {"layer": "", "module": "lm_head (tied embed_tokens)", "group": "output", "op": "linear", "K_in": H, "N_out": V,
         "weight_shape_unpacked": f"{V}x{H}", "weight_shape_packed_u8": "", "weights": V * H, "ternary": False, "has_scale": False,
         "scale_granularity": "", "dot_length": H, "dots_per_token": V, "repeats_per_block": "once per token"}])
    write_csv(os.path.join(out, "dot_length_summary.csv"), dot_rows)
    write_csv(os.path.join(out, "expected_terms_per_dot.csv"), expected_terms)
    write_csv(os.path.join(out, "operation_inventory.csv"), ops)
    write_csv(os.path.join(out, "memory_representation.csv"), mem_rows)
    write_csv(os.path.join(out, "decode_traffic_model.csv"), traffic)
    write_csv(os.path.join(out, "accumulator_widths.csv"), acc_rows)

    # ---- 12. Environment --------------------------------------------------------
    import numpy, huggingface_hub
    env = [
        "FlexTern EXP-0002 workload profile environment",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Command: python {argv}",
        f"Python: {sys.version.split()[0]} ({platform.platform()})",
        f"numpy: {numpy.__version__}   huggingface_hub: {huggingface_hub.__version__}",
        f"Model: {MODEL_ID}  revision {MODEL_REVISION}",
        f"Model file: {model_path}",
        f"File size: {layout['file_size']} (expected {EXPECTED_FILE_SIZE}: {'OK' if size_ok else 'MISMATCH'})",
        f"SHA-256: {file_sha}  (expected {EXPECTED_SHA256}: {'OK' if hash_ok else ('SKIPPED' if hash_ok is None else 'MISMATCH')})",
        f"Reference quantization code: transformers src/transformers/integrations/bitnet.py @ {TRANSFORMERS_BITNET_COMMIT}",
        "Random seeds: none (fully deterministic)",
        f"Runtime: {time.time() - t0:.1f} s",
    ]
    with open(os.path.join(out, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(env) + "\n")

    # ---- 13. Human-readable summary -------------------------------------------
    write_summary(out, model_summary, by_module, dot_rows, expected_terms, mem_rows, traffic, acc_rows, stats, by_layer, ops, ctx, dead_rows)
    print("\n".join(env))
    print(f"\nwrote outputs to {out}")
    return 0


def write_summary(out, ms, by_module, dot_rows, expected_terms, mem_rows, traffic, acc_rows, stats, by_layer, ops, ctx, dead_rows):
    g = ms["global_ternary_distribution"]; p = ms["parameters"]; o = ms["operation_summary"]; dc = ms["dead_channels"]
    lines = []
    A = lines.append
    A("# Workload profile — BitNet b1.58 2B4T (generated; do not hand-edit)")
    A("")
    A(f"Model `{ms['model_id']}` revision `{ms['revision'][:12]}`, {ms['license']}. Generated by `analysis/ternary_model_profile.py`; see `environment.txt`.")
    A("")
    A("Labels: MEASURED = counted from the checkpoint; INFERRED = computed from measured dimensions/format; HYPOTHETICAL = not what the model does.")
    A("")
    A("## 1. Model (from config.json, MEASURED)")
    A("")
    A("| Field | Value |"); A("|---|---|")
    for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "gqa_group_size", "vocab_size", "max_position_embeddings", "hidden_act", "tie_word_embeddings", "rope_theta", "rms_norm_eps"):
        A(f"| {k} | {ms[k]} |")
    A(f"| normalization | {ms['normalization']} |")
    A(f"| attention | {ms['attention']} |")
    A(f"| mlp | {ms['mlp']} |")
    A("")
    A("## 2. Parameters (MEASURED from tensor shapes)")
    A("")
    A("| Class | Count | % of total |"); A("|---|---:|---:|")
    A(f"| Ternary linear weights | {p['ternary_linear']:,} | {p['pct_ternary']:.1f}% |")
    A(f"| ├ attention projections (q,k,v,o) | {p['attention_proj_ternary']:,} | {100*p['attention_proj_ternary']/p['total']:.1f}% |")
    A(f"| └ MLP (gate,up,down) | {p['mlp_ternary']:,} | {100*p['mlp_ternary']/p['total']:.1f}% |")
    A(f"| Embedding BF16, tied as lm_head | {p['embedding_bf16_tied_lm_head']:,} | {p['pct_embedding']:.1f}% |")
    A(f"| Norm weights BF16 | {p['norm_bf16']:,} | {100*p['norm_bf16']/p['total']:.3f}% |")
    A(f"| Weight scales BF16 | {p['weight_scales_bf16']:,} | ~0% |")
    A(f"| **Total** | **{p['total']:,}** | 100% |")
    A("")
    A("## 3. Ternary weight distribution (MEASURED)")
    A("")
    A(f"Total ternary weights: **{g['total_ternary_weights']:,}**. Reserved 2-bit code 3 occurrences: {g['reserved_code3_count']}.")
    A("")
    A("| Value | Count | % |"); A("|---|---:|---:|")
    A(f"| -1 | {g['count_neg']:,} | {g['pct_neg']:.3f}% |")
    A(f"| 0 | {g['count_zero']:,} | {g['pct_zero']:.3f}% |")
    A(f"| +1 | {g['count_pos']:,} | {g['pct_pos']:.3f}% |")
    A(f"| sum | | {g['pct_sum']:.3f}% |")
    A("")
    A(f"Per-tensor zero rate: min {g['tensor_zero_rate_min_pct']:.2f}%, median {g['tensor_zero_rate_median_pct']:.2f}%, max {g['tensor_zero_rate_max_pct']:.2f}%, std {g['tensor_zero_rate_std_pct']:.2f} pts, size-weighted mean {g['tensor_zero_rate_size_weighted_mean_pct']:.2f}%.")
    A(f"Most sparse: `{g['most_sparse_tensor']}`. Least sparse: `{g['least_sparse_tensor']}`.")
    A(f"weight_scale range across all 210 tensors: {g['weight_scale_min']:.5f} .. {g['weight_scale_max']:.5f} (BF16, one per tensor, multiplies the output; equals absmean β).")
    A("")
    A("### By module type")
    A("")
    A("| Module | K | N | Weights | % of ternary | % zero | % -1 | % +1 | zero-rate range across layers | scale range |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for m in by_module:
        A(f"| {m['module']} | {m['K']} | {m['N']} | {m['weights']:,} | {m['pct_of_ternary']:.1f}% | {m['pct_zero']:.2f}% | {m['pct_neg']:.2f}% | {m['pct_pos']:.2f}% | {m['zero_rate_min_pct']:.1f}–{m['zero_rate_max_pct']:.1f}% | {m['scale_min']:.4f}–{m['scale_max']:.4f} |")
    A("")
    A("### By layer (zero %)")
    A("")
    A("| Layer | all | attention | MLP |"); A("|---:|---:|---:|---:|")
    for r in by_layer:
        A(f"| {r['layer']} | {r['pct_zero']:.1f} | {r['attn_pct_zero']:.1f} | {r['mlp_pct_zero']:.1f} |")
    A("")
    A("### Zero structure along K (MEASURED vs independence prediction)")
    A("")
    A("Fraction of aligned groups of g consecutive weights along the dot-product axis that are entirely zero, versus p^g if zeros were independent. Ratio ≈ 1 means no exploitable block structure.")
    A("")
    A("| Module | g=4 measured | g=4 p^4 | g=8 measured | g=8 p^8 | g=16 measured | g=16 p^16 | g=16 excl. all-zero rows | row zero-rate std (mean over tensors) | all-zero rows (sum) | all-zero cols (sum) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mod in [m["module"] for m in by_module]:
        ss = [s for s in stats if s.module == mod]
        w = np.array([s.numel for s in ss], dtype=float); w /= w.sum()
        def wm(attr): return float(sum(getattr(s, attr) * wi for s, wi in zip(ss, w)))
        A(f"| {mod} | {wm('g4_allzero_rate'):.4f} | {wm('g4_predicted'):.4f} | {wm('g8_allzero_rate'):.5f} | {wm('g8_predicted'):.5f} | {wm('g16_allzero_rate'):.6f} | {wm('g16_predicted'):.6f} | {wm('g16_allzero_rate_live_rows'):.6f} | {wm('row_zero_rate_std'):.4f} | {sum(s.allzero_rows for s in ss):,} | {sum(s.allzero_cols for s in ss):,} |")
    A("")
    A("### Dead channels (MEASURED)")
    A("")
    A(f"An all-zero weight row is an output channel that is identically zero for every input. In the MLP, a dead gate_proj or up_proj row makes that intermediate channel always zero, so the matching down_proj column never contributes. Dead MLP intermediate channels: **{dc['mlp_intermediate_channels_dead_total']:,} of {dc['mlp_intermediate_channels_total']:,} ({dc['pct_mlp_intermediate_channels_dead']:.2f}%)**. Ternary weights sitting in all-zero rows: {dc['pct_ternary_weights_in_dead_rows']:.2f}% of all ternary weights.")
    A("")
    A("| Layer | q dead rows | k | v | o | gate dead rows | up dead rows | gate∪up | gate∩up | down dead cols | % intermediate dead |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in dead_rows:
        A(f"| {r['layer']} | {r['q_dead_rows']} | {r['k_dead_rows']} | {r['v_dead_rows']} | {r['o_dead_rows']} | {r['gate_dead_rows']} | {r['up_dead_rows']} | {r['gate_or_up_dead']} | {r['gate_and_up_dead']} | {r['down_dead_cols']} | {r['pct_intermediate_channels_dead']:.2f} |")
    A("")
    A("## 4. Dot-product lengths (MEASURED shapes)")
    A("")
    A("| K | Modules | Tensors | Dot products per token (all layers) | Weights | % of ternary | % zero | Expected nonzero terms per dot |")
    A("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in dot_rows:
        A(f"| {r['K']} | {r['modules']} | {r['tensors']} | {r['dot_products_per_token']:,} | {r['weights']:,} | {r['pct_of_ternary_weights']:.1f}% | {r['pct_zero']:.2f}% | {r['expected_nonzero_terms']:,.0f} |")
    A("")
    A("Expected terms per representative dot product (INFERRED from measured rates):")
    A("")
    A("| Module | K | E[nonzero] | E[+1] | E[-1] | E[0] |"); A("|---|---:|---:|---:|---:|---:|")
    for e in expected_terms:
        A(f"| {e['module']} | {e['K']} | {e['expected_nonzero']:,.0f} | {e['expected_pos']:,.0f} | {e['expected_neg']:,.0f} | {e['expected_zero']:,.0f} |")
    A("")
    A("## 5. Activation and scale scheme (from reference code, KNOWN)")
    A("")
    A(f"- Weights: {ms['weight_quantization']}")
    A(f"- Activations: {ms['activation_quantization']}")
    A("- Scale granularity: weight scale per tensor (1 value per linear); activation scale per token per linear input (dynamic, computed at runtime).")
    A("- Scale application: one BF16 multiply per output element (weight_scale) and one divide per output element (activation scale), both after accumulation. Never inside the dot product.")
    A("")
    A(f"## 6. Operation mix per transformer block (INFERRED, static counts, context {ctx})")
    A("")
    A("| Operation | Class | Domain | Count/block | K | N | MACs or ops per token per block | Notes |")
    A("|---|---|---|---:|---:|---:|---:|---|")
    for r in ops:
        A(f"| {r['operation']} | {r['class']} | {r['domain']} | {r['count_per_block']} | {r['K']} | {r['N']} | {r['macs_or_ops_per_token_per_block']:,} | {r['notes']} |")
    A("")
    A(f"Totals per token: ternary linear MACs {o['ternary_macs_per_token_total']:,} ({o['pct_linear_macs_ternary']:.1f}% of linear MACs); tied BF16 lm_head MACs {o['lm_head_bf16_macs_per_token']:,} ({o['pct_linear_macs_lm_head_bf16']:.1f}%); attention score MACs at context {ctx}: {o['attention_score_macs_per_token_total_at_context']:,}.")
    A("")
    A("## 7. Weight storage (MEASURED file / THEORETICAL / HYPOTHETICAL)")
    A("")
    A("| Representation | Scope | Bytes | Note |"); A("|---|---|---:|---|")
    for r in mem_rows:
        A(f"| {r['representation']} | {r['scope']} | {fmt_bytes(r['bytes'])} | {r['note']} |")
    A("")
    A(f"## 8. Batch-1 decode weight traffic (INFERRED / ANALYTICAL MODEL, context {ctx})")
    A("")
    A("Assumes every weight byte is streamed from memory exactly once per generated token, no on-chip residency, no batching, no compression. KV-cache read column assumes the full BF16 cache at the stated context is read once per token.")
    A("")
    A("| Scenario | Ternary layers | Head | Norms+scales | Weight bytes/token | % head | KV read/token | Total/token |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in traffic:
        A(f"| {r['scenario']} | {fmt_bytes(r['ternary_layer_bytes_per_token'])} | {fmt_bytes(r['head_bytes_per_token'])} | {fmt_bytes(r['norm_and_scale_bytes_per_token'])} | {fmt_bytes(r['weight_bytes_per_token'])} | {r['pct_head']:.0f}% | {fmt_bytes(r['kv_cache_read_bytes_per_token_at_context'])} | {fmt_bytes(r['total_bytes_per_token_at_context'])} |")
    A("")
    A("Invalidated by: on-chip SRAM residency of any part of the model, batching (weights amortized over tokens), weight reuse across heads/queries, compression, multi-chip sharding, speculative decoding, or a quantized head.")
    A("")
    A("## 9. Accumulator widths (INFERRED mathematically)")
    A("")
    A("| K | Modules | Format range | Format sum range | Format bits | Model range | Model sum range | Model bits | Max nonzero/row (data) | Data-bounded bits |")
    A("|---:|---|---|---|---:|---|---|---:|---:|---:|")
    for r in acc_rows:
        A(f"| {r['K']} | {r['modules']} | {r['format_range']} | [{r['format_sum_min']:,}, {r['format_sum_max']:,}] | {r['format_signed_bits']} | {r['model_range']} | [{r['model_sum_min']:,}, {r['model_sum_max']:,}] | {r['model_signed_bits']} | {r['max_nonzero_in_any_row']} | {r['data_bounded_signed_bits']} |")
    A("")
    A("Format = INT8 activations [-128,127] where ternary -1 turns -128 into +128 (the Phase 1 Y-Core case). Model = what activation_quant can actually emit ([-127,127]). Data-bounded = using the largest nonzero count in any weight row of this checkpoint; it is a property of this file, not a guarantee.")
    A("")
    with open(os.path.join(out, "workload_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
