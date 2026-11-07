"""ANALYTICAL per-control-step cost model (MACs and weight/state bytes). Nothing here is measured.

Bit widths used for weight bytes: fp = 16 (FP16), int8 = 8, int4 = 4 (+ 2-byte scale per output channel), tern = 2
(+ one 2-byte scale per tensor). Layers outside the quantization scope, and the first conv, stay FP16. The recurrent
state is read and written once per step at its stored width (16 bits, or `state_bits`). Activation traffic and on-chip
reuse are NOT modelled (every weight byte is assumed fetched once per step: the batch-1 decode regime).
A ternary 'MAC' is an add/subtract/skip; counts are reported per layer so the claim stays inspectable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import env as E
from . import models as M
from .quant import QConv2d, QLinear

BITS = {"fp": 16, "int8": 8, "int4": 4, "tern": 2}


def _component(name):
    for p in ("enc", "cell", "pol", "pred", "trig", "inp"):
        if name.startswith(p):
            return p
    return "other"


def layer_table(arm):
    """Per-layer rows: component, name, kind, mode, params, macs (one forward, batch 1, one control step)."""
    net = M.build(arm)
    rows = []
    hooks = []

    def mk(name):
        def hook(mod, inp, out):
            if isinstance(mod, (nn.Conv2d,)):
                macs = out.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
                params = mod.weight.numel()
            else:
                macs = mod.in_features * mod.out_features
                params = mod.weight.numel()
            mode = getattr(mod, "mode", "fp")
            rows.append({"component": _component(name), "name": name, "mode": mode, "params": params, "macs": macs,
                         "out_ch": mod.weight.shape[0]})
        return hook

    for name, m in net.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and not name.startswith("tf") and not name.startswith("ema"):
            hooks.append(m.register_forward_hook(mk(name)))
    c = net.cfg
    with torch.no_grad():
        extra = 1 + 4 + E.N_COLORS + 3
        if c.kind == "stack":
            z = net.enc(torch.zeros(1, 12, 32, 32))
            net.pol(torch.zeros(1, M.Z + extra + 3))
        elif c.kind == "react":
            z = net.enc(torch.zeros(1, 3, 32, 32))
            net.pol(torch.zeros(1, M.Z + extra))
        elif c.kind == "trans":
            z = net.enc(torch.zeros(1, 3, 32, 32))
            net.inp(torch.zeros(1, M.Z + extra))
            net.pol(torch.zeros(1, 128 + 4 + E.N_COLORS))
        else:
            z = net.enc(torch.zeros(1, 3, 32, 32))
            h = torch.zeros(1, M.H)
            net.cell_step(h, z, torch.ones(1), torch.zeros(1, 4), torch.zeros(1, E.N_COLORS), torch.zeros(1, 3))
            net.policy(h, torch.zeros(1, 4), torch.zeros(1, E.N_COLORS))
            net.pred(torch.zeros(1, M.H + 3))
            net.trig(torch.zeros(1, M.H + 3 + M.POOL_DIM))
    for h in hooks:
        h.remove()
    if c.kind == "trans":                         # transformer blocks (KV-cached): analytical, L=16 window
        for i in range(2):
            rows.append({"component": "tf", "name": f"tf.{i}.qkvo+ff", "mode": "fp",
                         "params": 4 * 128 * 128 + 2 * 128 * 256, "macs": 4 * 128 * 128 + 2 * 128 * 256 + 2 * 16 * 128,
                         "out_ch": 1})
    return rows


def weight_bytes(row, arm_cfg):
    mode = row["mode"]
    if row["name"] in ("enc.c1",):
        mode = "fp"
    bits = BITS[mode]
    extra = 2 * row["out_ch"] if mode == "int4" or mode == "int8" else (2 if mode == "tern" else 0)
    return row["params"] * bits / 8 + extra


def step_cost(arm, enc_frac=1.0, trigger=False):
    """Average per-control-step cost when the encoder runs on a fraction `enc_frac` of steps.
    trigger=True charges the trigger head (always executed) as well; the 8x8 pooling is charged as POOL_DIM*... adds."""
    cfg = M.ARMS[arm]
    rows = layer_table(arm)
    comp = {}
    for r in rows:
        d = comp.setdefault(r["component"], {"macs": 0, "bytes": 0.0, "params": 0})
        d["macs"] += r["macs"]
        d["bytes"] += weight_bytes(r, cfg)
        d["params"] += r["params"]
    state_bytes = 0.0
    if cfg.kind == "rec":
        sb = (cfg.state_bits or 16) / 8
        state_bytes = 2 * M.H * sb
    enc = comp.get("enc", {"macs": 0, "bytes": 0.0})
    always_macs = sum(v["macs"] for k, v in comp.items() if k not in ("enc", "trig"))
    always_bytes = sum(v["bytes"] for k, v in comp.items() if k not in ("enc", "trig")) + state_bytes
    if cfg.kind == "rec" and not cfg.jepa:
        pass
    # the predictor head is training/analysis-only unless used for planning: exclude from the deployed per-step cost
    pred = comp.get("pred", {"macs": 0, "bytes": 0.0})
    always_macs -= pred["macs"]
    always_bytes -= pred["bytes"]
    trig = comp.get("trig", {"macs": 0, "bytes": 0.0})
    t_macs = trig["macs"] + 3 * E.RES * E.RES if trigger else 0           # + pooling adds over 32x32x3 pixels
    t_bytes = trig["bytes"] if trigger else 0.0
    return {"arm": arm, "enc_frac": enc_frac,
            "macs": always_macs + enc_frac * enc["macs"] + t_macs,
            "bytes": always_bytes + enc_frac * enc["bytes"] + t_bytes,
            "enc_macs": enc["macs"], "enc_bytes": enc["bytes"], "state_bytes": state_bytes,
            "updater_macs": always_macs, "updater_bytes": always_bytes,
            "params": sum(v["params"] for v in comp.values()), "components": comp}


def amdahl(arm):
    """Fraction of per-step MACs/bytes at full perception that is encoder (so non-encoder savings are bounded)."""
    c = step_cost(arm, 1.0)
    return {"enc_share_macs": c["enc_macs"] / c["macs"], "enc_share_bytes": c["enc_bytes"] / c["bytes"]}
