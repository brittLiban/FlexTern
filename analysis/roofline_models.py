"""
FlexTern EXP-0003 — analytical compute-vs-memory bottleneck model (library).

Everything here is ANALYTICAL / INFERRED. Workload quantities come from the
canonical EXP-0002/EXP-0002B profiles (MEASURED); hardware capabilities are
abstract assumptions with no measurement behind them. No number produced
here is a hardware result.

Units: bytes, seconds, operations. Bandwidth in bytes/s (decimal GB/s, TB/s).

Per generated token at batch B (autoregressive decode, context C = current KV length):

  Ternary linears   compute: W terms (dense) per token; weight bytes W*bits/8 streamed once per
                    step and shared by the B sequences, so W*bits/8/B per token.
  Output head       compute: V*H MACs per token; bytes V*H*b_head/B per token.
  Attention         compute: L*(2*nH*hd*C + 4*nH*C) op-equivalents per token (QK^T, AV, softmax);
                    memory: KV read 2*nKV*hd*b_kv*L*C per token (sequence-specific, not shared),
                    KV write 2*nKV*hd*b_kv*L per token.
  Other             compute: norms, activation quantization, scale application, RoPE, residual adds,
                    activation function, gating; memory: norm and scale parameters / B.
  Activations are assumed on-chip and are reported as counts only.

Overlap bounds per stage i with compute_i and memory_i:
  optimistic   T = sum_i max(compute_i, memory_i)   (within-stage overlap only)
  conservative T = sum_i (compute_i + memory_i)
No cross-stage prefetch is assumed in either bound.

Ternary-only Amdahl ceilings (on the optimistic bound unless stated):
  compute-only elimination: ternary stage time -> its memory time
  bandwidth-only elimination: ternary stage time -> its compute time
  ideal elimination: ternary stage time -> 0
  ceiling = T_total / T_modified

Memory-bound identity: when every stage is memory-bound, T = bytes/BW and the ideal
ceiling equals total_bytes / non_ternary_bytes, independent of compute rates.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace

# ----------------------------------------------------------------------------
# Workload (MEASURED inputs from the canonical profiles)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    slug: str
    model_id: str
    revision: str
    L: int
    H: int
    I: int
    nH: int
    nKV: int
    hd: int
    V: int
    max_ctx: int
    tied: bool
    W: int                      # ternary weights (= dense ternary terms per token)
    zero_rate: float            # fraction
    norm_params: int
    scale_params: int
    head_params: int
    head_bytes_measured: int    # BF16 bytes of the head tensor as stored
    norm_names: tuple[str, ...]
    hidden_act: str
    modules: tuple[dict, ...]   # per projection: module, K, N, weights
    source_dir: str

    @classmethod
    def from_profile_dir(cls, d: str) -> "Workload":
        with open(os.path.join(d, "model_summary.json"), encoding="utf-8") as f:
            ms = json.load(f)
        with open(os.path.join(d, "weight_distribution_by_module.csv"), encoding="utf-8") as f:
            mods = tuple({"module": r["module"], "K": int(r["K"]), "N": int(r["N"]), "weights": int(r["weights"]),
                          "pct_zero": float(r["pct_zero"])} for r in csv.DictReader(f))
        p = ms["parameters"]
        return cls(
            slug=os.path.basename(d.rstrip("/\\")) if "falcon" in d else "bitnet-2b4t",
            model_id=ms["model_id"], revision=ms["revision"],
            L=ms["num_hidden_layers"], H=ms["hidden_size"], I=ms["intermediate_size"],
            nH=ms["num_attention_heads"], nKV=ms["num_key_value_heads"], hd=ms["head_dim"],
            V=ms["vocab_size"], max_ctx=ms["max_position_embeddings"], tied=bool(ms["tie_word_embeddings"]),
            W=p["ternary_linear"], zero_rate=ms["global_ternary_distribution"]["pct_zero"] / 100.0,
            norm_params=p["norm_bf16"], scale_params=p["weight_scales_bf16"],
            head_params=ms["head"]["params"], head_bytes_measured=ms["head"]["bytes"],
            norm_names=tuple(ms["norm_names"]), hidden_act=ms["hidden_act"], modules=mods, source_dir=d,
        )

    # ---- derived MEASURED quantities ----
    @property
    def terms_per_token(self) -> int:
        return self.W

    @property
    def norm_elems_per_layer(self) -> int:
        return sum(self.I if "ffn_sub_norm" in n else self.H for n in self.norm_names)

    @property
    def outputs_per_layer(self) -> int:
        return sum(m["N"] for m in self.modules)

    @property
    def activation_inputs_per_layer(self) -> int:
        """Distinct quantized input vectors per layer: pre-qkv (H), pre-o (nH*hd), pre-gate/up (H), pre-down (I)."""
        return self.H + self.nH * self.hd + self.H + self.I


# ----------------------------------------------------------------------------
# Abstract hardware / scenario assumptions (NOT measurements)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Assumptions:
    bw: float = 250e9                  # external memory bandwidth, bytes/s
    terms_per_cycle: int = 4096        # abstract ternary engine width
    clock_hz: float = 1e9
    nonternary_rate: float = 1e12      # abstract mixed-precision op-equivalents/s (no evidence; swept)
    weight_bits: float = 2.0           # 2.0 (MEASURED packing) or 1.6 (HYPOTHETICAL 5 trits/byte)
    act_bits: int = 8                  # 8 MEASURED; 4 HYPOTHETICAL
    act_bitplane_scaling: bool = False  # HYPOTHETICAL: if True, ternary rate scales by 8/act_bits
    head_bytes_per_elem: float = 2.0   # 2 BF16 MEASURED; 1 INT8 / 0.5 INT4 HYPOTHETICAL
    kv_bytes_per_elem: float = 2.0     # 2 BF16 reference; 1 / 0.5 HYPOTHETICAL
    zero_mode: str = "dense"           # dense | gated | ideal_skip
    batch: int = 1
    context: int = 4096
    resident_bytes_weights: float = 0.0   # external weight bytes removed by residency (per step)
    kv_resident: bool = False             # HYPOTHETICAL: KV held on-chip

    @property
    def ternary_rate(self) -> float:
        r = self.terms_per_cycle * self.clock_hz
        if self.act_bitplane_scaling:
            r *= 8.0 / self.act_bits
        return r

    @property
    def terms_per_weight_byte(self) -> float:
        return 8.0 / self.weight_bits


# ----------------------------------------------------------------------------
# Component equations
# ----------------------------------------------------------------------------


def packed_weight_bytes(w: Workload, weight_bits: float) -> float:
    """Dense packed ternary bytes for the whole model. Zeros still occupy bits."""
    return w.W * weight_bits / 8.0


def head_bytes(w: Workload, bytes_per_elem: float) -> float:
    return w.V * w.H * bytes_per_elem


def norm_scale_bytes(w: Workload) -> float:
    return 2.0 * (w.norm_params + w.scale_params)


def kv_bytes_per_token_per_ctx(w: Workload, kv_bytes_per_elem: float) -> float:
    """KV read bytes per generated token per unit of context (K and V, all KV heads, all layers)."""
    return 2.0 * w.nKV * w.hd * kv_bytes_per_elem * w.L


def kv_read_bytes(w: Workload, ctx: int, kv_bytes_per_elem: float) -> float:
    return kv_bytes_per_token_per_ctx(w, kv_bytes_per_elem) * ctx


def kv_write_bytes(w: Workload, kv_bytes_per_elem: float) -> float:
    return 2.0 * w.nKV * w.hd * kv_bytes_per_elem * w.L


def attention_ops(w: Workload, ctx: int) -> float:
    """Decode attention op-equivalents per token: QK^T + AV (MACs) and softmax (~4 ops/score)."""
    return w.L * (2.0 * w.nH * w.hd * ctx + 4.0 * w.nH * ctx)


def head_ops(w: Workload) -> float:
    return float(w.V * w.H)


def other_ops(w: Workload) -> float:
    """Per token: norms (~3 ops/elem), activation quant (absmax + scale/round = 2 ops/elem),
    scale application (1/output), RoPE (~2 ops/elem on q,k), residual adds, activation fn, gating."""
    per_layer = (3.0 * w.norm_elems_per_layer
                 + 2.0 * w.activation_inputs_per_layer
                 + 1.0 * w.outputs_per_layer
                 + 2.0 * (w.nH + w.nKV) * w.hd
                 + 2.0 * w.H
                 + 1.0 * w.I
                 + 1.0 * w.I)
    return w.L * per_layer + 3.0 * w.H  # + final norm


def ternary_compute_terms(w: Workload, zero_mode: str) -> float:
    if zero_mode == "dense":
        return float(w.W)
    if zero_mode == "gated":        # engine cycles unchanged; useful terms fewer (reported separately)
        return float(w.W)
    if zero_mode == "ideal_skip":   # upper bound: only nonzero terms consume engine cycles
        return w.W * (1.0 - w.zero_rate)
    raise ValueError(zero_mode)


def activation_bytes_per_token(w: Workload, act_bits: int) -> float:
    """On-chip activation bytes quantized per token (count only; not external traffic)."""
    return w.L * w.activation_inputs_per_layer * act_bits / 8.0


# ----------------------------------------------------------------------------
# Stage model
# ----------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    compute_ops: float
    memory_bytes: float
    compute_time: float
    memory_time: float

    @property
    def optimistic(self) -> float:
        return max(self.compute_time, self.memory_time)

    @property
    def conservative(self) -> float:
        return self.compute_time + self.memory_time

    @property
    def limiter(self) -> str:
        return "memory" if self.memory_time >= self.compute_time else "compute"


@dataclass
class TokenModel:
    workload: str
    assumptions: Assumptions
    stages: list[Stage]
    hypothetical_context: bool

    def stage(self, name: str) -> Stage:
        return next(s for s in self.stages if s.name == name)

    @property
    def total_optimistic(self) -> float:
        return sum(s.optimistic for s in self.stages)

    @property
    def total_conservative(self) -> float:
        return sum(s.conservative for s in self.stages)

    @property
    def total_bytes(self) -> float:
        return sum(s.memory_bytes for s in self.stages)

    @property
    def total_ops(self) -> float:
        return sum(s.compute_ops for s in self.stages)

    @property
    def nonternary_bytes(self) -> float:
        return self.total_bytes - self.stage("ternary").memory_bytes

    @property
    def limiting_component(self) -> str:
        return max(self.stages, key=lambda s: s.optimistic).name

    @property
    def ternary_time_fraction(self) -> float:
        t = self.total_optimistic
        return self.stage("ternary").optimistic / t if t else 0.0

    @property
    def ternary_bytes_fraction(self) -> float:
        b = self.total_bytes
        return self.stage("ternary").memory_bytes / b if b else 0.0

    def ceilings(self) -> dict:
        """Amdahl ceilings on the optimistic bound."""
        t = self.stage("ternary")
        T = self.total_optimistic
        rest = T - t.optimistic
        return {
            "ceiling_compute_only": T / (rest + t.memory_time) if (rest + t.memory_time) > 0 else math.inf,
            "ceiling_bandwidth_only": T / (rest + t.compute_time) if (rest + t.compute_time) > 0 else math.inf,
            "ceiling_ideal": T / rest if rest > 0 else math.inf,
            "ceiling_ideal_conservative": self.total_conservative / (self.total_conservative - t.conservative)
            if (self.total_conservative - t.conservative) > 0 else math.inf,
            "memory_bound_bytes_ratio": self.total_bytes / self.nonternary_bytes if self.nonternary_bytes > 0 else math.inf,
        }

    def tokens_per_second_optimistic(self) -> float:
        """ANALYTICAL / MODELED THROUGHPUT UNDER STATED ASSUMPTIONS. Never a measurement."""
        return 1.0 / self.total_optimistic if self.total_optimistic > 0 else math.inf


def token_model(w: Workload, a: Assumptions) -> TokenModel:
    B = a.batch
    C = a.context
    # --- ternary linears
    t_ops = ternary_compute_terms(w, a.zero_mode)
    w_bytes_step = packed_weight_bytes(w, a.weight_bits)
    # residency removes external weight bytes (head first is handled by caller via resident_bytes_weights split)
    t_bytes_step = max(0.0, w_bytes_step - a.resident_bytes_weights) if a.resident_bytes_weights <= w_bytes_step else 0.0
    t_bytes = t_bytes_step / B
    ternary = Stage("ternary", t_ops, t_bytes, t_ops / a.ternary_rate, t_bytes / a.bw)
    # --- output head (weights shared across batch)
    h_bytes_step = head_bytes(w, a.head_bytes_per_elem)
    spill = max(0.0, a.resident_bytes_weights - w_bytes_step)   # residency beyond ternary spills to head
    h_bytes = max(0.0, h_bytes_step - spill) / B
    head = Stage("head", head_ops(w), h_bytes, head_ops(w) / a.nonternary_rate, h_bytes / a.bw)
    # --- attention + KV (sequence-specific; no batch sharing)
    kv_b = 0.0 if a.kv_resident else (kv_read_bytes(w, C, a.kv_bytes_per_elem) + kv_write_bytes(w, a.kv_bytes_per_elem))
    attn = Stage("attention_kv", attention_ops(w, C), kv_b, attention_ops(w, C) / a.nonternary_rate, kv_b / a.bw)
    # --- other
    o_bytes = norm_scale_bytes(w) / B
    other = Stage("other", other_ops(w), o_bytes, other_ops(w) / a.nonternary_rate, o_bytes / a.bw)
    return TokenModel(w.slug, a, [ternary, head, attn, other], hypothetical_context=C > w.max_ctx)


# ----------------------------------------------------------------------------
# Analytical crossovers and break-even equations
# ----------------------------------------------------------------------------


def context_crossovers(w: Workload, a: Assumptions) -> dict:
    """Context lengths (per-token accounting at batch B) where KV read bytes cross weight streams.
    Negative numerators mean the condition holds at every context ('always')."""
    B = a.batch
    kv1 = kv_bytes_per_token_per_ctx(w, a.kv_bytes_per_elem)
    tern = packed_weight_bytes(w, a.weight_bits) / B
    head = head_bytes(w, a.head_bytes_per_elem) / B
    norms = norm_scale_bytes(w) / B

    def cx(num):
        return 0.0 if num <= 0 else num / kv1

    return {
        "ctx_kv_gt_ternary": cx(tern),
        "ctx_kv_gt_head": cx(head),
        "ctx_kv_gt_all_weights": cx(tern + head + norms),
        "ctx_kv_plus_head_gt_ternary": cx(tern - head),
        "ctx_nonternary_mem_gt_ternary_mem": cx(tern - head - norms),      # ideal memory-bound ceiling = 2x boundary
        "ctx_ceiling_3x_memory_bound": cx(tern / 2.0 - head - norms),      # ternary bytes = 2 x non-ternary bytes
        "ctx_ceiling_1p5x_memory_bound": cx(2.0 * tern - head - norms),    # ternary bytes = 0.5 x non-ternary bytes
        "kv_bytes_per_token_per_ctx": kv1,
    }


def break_even(w: Workload, a: Assumptions) -> dict:
    """Ternary stage compute-vs-memory balance (per token at batch B)."""
    tpb = a.terms_per_weight_byte
    return {
        "terms_per_s_to_saturate_bw": a.bw * tpb * a.batch,            # rate at which compute time == memory time
        "bw_to_feed_engine": a.ternary_rate / (tpb * a.batch),
        "lanes_at_bw_and_clock": a.bw * tpb * a.batch / a.clock_hz,
        "batch_to_balance_ternary": a.ternary_rate / (tpb * a.bw),     # B* where compute == memory for ternary stage
    }


def batch_for_weight_eq_kv(w: Workload, a: Assumptions) -> float:
    """Batch at which per-token streamed weight bytes (ternary + head + norms) equal per-token KV read bytes at context."""
    kv = kv_read_bytes(w, a.context, a.kv_bytes_per_elem)
    weights = packed_weight_bytes(w, a.weight_bits) + head_bytes(w, a.head_bytes_per_elem) + norm_scale_bytes(w)
    return weights / kv if kv > 0 else math.inf


# ----------------------------------------------------------------------------
# Residency
# ----------------------------------------------------------------------------


def storage_facts(w: Workload, weight_bits: float = 2.0) -> dict:
    per_layer = sum(m["weights"] for m in w.modules) / w.L * weight_bits / 8.0   # module weights are model totals
    tensor_bytes = [m["weights"] / w.L * weight_bits / 8.0 for m in w.modules]
    return {
        "one_layer_bytes": per_layer,
        "largest_tensor_bytes": max(tensor_bytes),
        "smallest_tensor_bytes": min(tensor_bytes),
        "head_bytes_bf16": head_bytes(w, 2.0),
        "full_ternary_bytes": packed_weight_bytes(w, weight_bits),
        "all_weights_bytes": packed_weight_bytes(w, weight_bits) + head_bytes(w, 2.0) + norm_scale_bytes(w),
    }


def residency_external_bytes(w: Workload, a: Assumptions, capacity: float, policy: str) -> dict:
    """External bytes per token under a residency policy. No hit rates are invented."""
    tern = packed_weight_bytes(w, a.weight_bits)
    head = head_bytes(w, a.head_bytes_per_elem)
    norms = norm_scale_bytes(w)
    kv = kv_read_bytes(w, a.context, a.kv_bytes_per_elem) + kv_write_bytes(w, a.kv_bytes_per_elem)
    B = a.batch
    layer = tern / w.L
    rt = rh = rk = rn = 0.0
    if policy == "none":
        pass
    elif policy == "head_if_fits":
        rh = head if head <= capacity else 0.0
    elif policy == "one_layer_buffer":
        pass                            # a working buffer does not remove stream-once traffic
    elif policy == "greedy_ternary_layers":
        n = min(w.L, int(capacity // layer)) if layer > 0 else 0
        rt = n * layer
    elif policy == "greedy_head_then_layers":
        rh = head if head <= capacity else 0.0
        n = min(w.L, int((capacity - rh) // layer)) if layer > 0 else 0
        rt = n * layer
    elif policy == "ideal_ternary_first":
        # any weight byte resident saves one external byte per step; ternary filled first
        r = min(capacity, tern + head + norms)
        rt = min(r, tern); r -= rt
        rh = min(r, head); r -= rh
        rn = min(r, norms)
    elif policy == "ideal_nonternary_first":
        # same capacity, non-ternary weight streams filled first (raises the ternary-only ceiling most)
        r = min(capacity, tern + head + norms)
        rh = min(r, head); r -= rh
        rn = min(r, norms); r -= rn
        rt = min(r, tern)
    elif policy == "ideal_including_kv":
        # HYPOTHETICAL: KV cache at this context may also be held on-chip; non-ternary first
        r = min(capacity, tern + head + norms + kv)
        rh = min(r, head); r -= rh
        rn = min(r, norms); r -= rn
        rk = min(r, kv); r -= rk
        rt = min(r, tern)
    else:
        raise ValueError(policy)
    ext_tern = (tern - rt) / B
    ext_head = (head - rh) / B
    ext_kv = kv - rk
    ext_norms = (norms - rn) / B
    return {"policy": policy, "capacity_bytes": capacity, "resident_ternary": rt, "resident_head": rh, "resident_kv": rk,
            "resident_norms": rn,
            "external_ternary_bytes": ext_tern, "external_head_bytes": ext_head, "external_kv_bytes": ext_kv,
            "external_other_bytes": ext_norms, "external_total_bytes": ext_tern + ext_head + ext_kv + ext_norms,
            "ideal_ceiling_memory_bound": (ext_tern + ext_head + ext_kv + ext_norms) / (ext_head + ext_kv + ext_norms)
            if (ext_head + ext_kv + ext_norms) > 0 else math.inf}


# ----------------------------------------------------------------------------
# Tiles, accumulators, prefill
# ----------------------------------------------------------------------------


def tile_utilization(K: int, T: int) -> dict:
    tiles = math.ceil(K / T)
    return {"K": K, "tile": T, "tiles": tiles, "tail_utilization": K / (tiles * T), "partial_sums": tiles,
            "divisible": K % T == 0}


def accumulator_bits(K: int, act_bits: int) -> dict:
    """Signed accumulator bits for K ternary terms with act_bits activations.
    format range: [-2^(b-1), 2^(b-1)-1]; model range (absmax quantizer, -2^(b-1) unreachable): symmetric."""
    fmt_max = 1 << (act_bits - 1)          # |min| dominates via w = -1
    mdl_max = (1 << (act_bits - 1)) - 1

    def bits(mag):
        b = 1
        while (1 << (b - 1)) - 1 < mag:
            b += 1
        return b
    return {"K": K, "act_bits": act_bits, "format_sum_max": K * fmt_max, "format_bits": bits(K * fmt_max),
            "model_sum_max": K * mdl_max, "model_bits": bits(K * mdl_max)}


def prefill_attention_equals_ternary(w: Workload) -> float:
    """Prompt length P at which causal attention compute (~nH*hd*L*P^2) equals ternary compute (W*P)."""
    return w.W / (w.nH * w.hd * w.L)


def prefill_model(w: Workload, a: Assumptions, P: int) -> dict:
    """Prefill of a P-token prompt: weights stream once for all P tokens; attention is causal (avg context P/2)."""
    t_ops = w.W * P
    t_bytes = packed_weight_bytes(w, a.weight_bits)
    attn_ops = w.L * (2.0 * w.nH * w.hd * P * (P + 1) / 2.0 + 4.0 * w.nH * P * (P + 1) / 2.0)
    kv_w = kv_write_bytes(w, a.kv_bytes_per_elem) * P
    head_o = head_ops(w)  # only the last position needs logits for generation
    hb = head_bytes(w, a.head_bytes_per_elem)
    t = {"ternary": max(t_ops / a.ternary_rate, t_bytes / a.bw), "attention": max(attn_ops / a.nonternary_rate, kv_w / a.bw),
         "head": max(head_o / a.nonternary_rate, hb / a.bw), "other": other_ops(w) * P / a.nonternary_rate}
    return {"P": P, "ternary_ops": t_ops, "ternary_bytes": t_bytes, "ternary_bytes_per_token": t_bytes / P,
            "attention_ops": attn_ops, "kv_write_bytes": kv_w, "ternary_compute_time": t_ops / a.ternary_rate,
            "ternary_memory_time": t_bytes / a.bw, "ternary_limiter": "compute" if t_ops / a.ternary_rate >= t_bytes / a.bw else "memory",
            "attention_time": t["attention"], "total_optimistic": sum(t.values()), "ternary_fraction": t["ternary"] / sum(t.values())}


def assumptions_dict(a: Assumptions) -> dict:
    d = asdict(a)
    d["ternary_rate"] = a.ternary_rate
    d["terms_per_weight_byte"] = a.terms_per_weight_byte
    return d
