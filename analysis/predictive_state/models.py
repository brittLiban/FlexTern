"""EXP-0012 models. One factory, `build(arm)`, returns a Net for every arm in ARMS.

Recurrent arms: Encoder -> GRU state updater -> {policy head, action-conditioned latent predictor, trigger head}.
Non-recurrent controls: R (single frame), S4 (4-frame stack), T16 (causal transformer over 16 latents).
JEPA objective (arms with jepa=True): predictor(h_t, a_t) regresses the EMA target-encoder embedding of o_{t+1}
(stop-gradient, normalized, cosine loss). It is an auxiliary loss on top of behaviour cloning, not the policy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import env as E
from .quant import QConv2d, QLinear, fq_state

Z = 128
H = 128
POOL = 8
POOL_DIM = 3 * POOL * POOL


@dataclass(frozen=True)
class ArmCfg:
    kind: str            # rec | react | stack | trans
    wmode: str = "fp"    # fp | int8 | int4 | tern
    scope: str = "none"  # none | updater | all
    jepa: bool = True
    state_bits: int | None = None


ARMS = {
    "R": ArmCfg("react", jepa=False),
    "S4": ArmCfg("stack", jepa=False),
    "T16": ArmCfg("trans", jepa=False),
    "G0": ArmCfg("rec", jepa=False),
    "G": ArmCfg("rec"),
    "G-I8u": ArmCfg("rec", "int8", "updater"),
    "G-I8a": ArmCfg("rec", "int8", "all"),
    "G-I4u": ArmCfg("rec", "int4", "updater"),
    "G-I4a": ArmCfg("rec", "int4", "all"),
    "G-Tu": ArmCfg("rec", "tern", "updater"),
    "G-Ta": ArmCfg("rec", "tern", "all"),
    "G-Ta-s8": ArmCfg("rec", "tern", "all", state_bits=8),
}


class Encoder(nn.Module):
    """32x32x3 -> Z, conditioned on the task code (colour one-hot broadcast as channels) and CoordConv x/y channels
    (amendment 1: the unconditioned encoder could not localize the target, see the experiment record).
    First conv stays full precision; the rest follows `mode`."""

    def __init__(self, in_ch=3, mode="fp"):
        super().__init__()
        self.c1 = QConv2d(in_ch + E.N_COLORS + 2, 16, 3, 2, 1, "fp")
        self.c2 = QConv2d(16, 32, 3, 2, 1, mode)
        self.c3 = QConv2d(32, 64, 3, 2, 1, mode)
        self.fc = QLinear(1024, Z, mode)
        self.ln = nn.LayerNorm(Z)
        c = (torch.arange(E.RES).float() + 0.5) / E.RES
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None], persistent=False)

    def forward(self, x, task):               # x (N, C, 32, 32) in [-0.5, 0.5]; task (N, N_COLORS)
        n = x.shape[0]
        x = torch.cat([x, task[:, :, None, None].expand(-1, -1, E.RES, E.RES), self.coords.expand(n, -1, -1, -1)], 1)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return self.ln(self.fc(x.flatten(1)))


class GRUCell(nn.Module):
    def __init__(self, i, h, mode):
        super().__init__()
        self.wi = QLinear(i, 3 * h, mode)
        self.wh = QLinear(h, 3 * h, mode)

    def forward(self, x, h):
        gi, gh = self.wi(x).chunk(3, -1), self.wh(h).chunk(3, -1)
        r = torch.sigmoid(gi[0] + gh[0])
        u = torch.sigmoid(gi[1] + gh[1])
        n = torch.tanh(gi[2] + r * gh[2])
        return (1 - u) * n + u * h


def mlp(i, hdim, o, mode):
    return nn.Sequential(QLinear(i, hdim, mode), nn.ReLU(), QLinear(hdim, o, "fp"))


def to_x(img_u8):
    """(..., 32, 32, 3) uint8 -> (N, 3, 32, 32) float."""
    s = img_u8.shape
    x = img_u8.reshape(-1, *s[-3:]).permute(0, 3, 1, 2).float() / 255.0 - 0.5
    return x


def pool_feat(img_u8):
    """cheap sentinel features: 8x8 average pool of the raw frame -> (..., POOL_DIM)."""
    s = img_u8.shape
    x = to_x(img_u8)
    return F.adaptive_avg_pool2d(x, POOL).flatten(1).reshape(*s[:-3], POOL_DIM)


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        c = ARMS[arm]
        self.cfg = c
        enc_mode = c.wmode if c.scope == "all" else "fp"
        up_mode = c.wmode if c.scope in ("updater", "all") else "fp"
        in_ch = 12 if c.kind == "stack" else 3
        self.enc = Encoder(in_ch, enc_mode)
        extra = 1 + 4 + E.N_COLORS + 3                       # valid, prop, task, a_prev
        if c.kind == "rec":
            self.cell = GRUCell(Z + extra, H, up_mode)
            self.pol = mlp(H + 4 + E.N_COLORS + Z, 128, 3, up_mode)   # state + prop + task + current (masked) embedding
            self.pred = mlp(H + 3, 128, Z, up_mode)
            self.trig = mlp(H + 3 + POOL_DIM, 64, 1, up_mode)
        elif c.kind in ("react", "stack"):
            self.pol = mlp(Z + extra + (3 if c.kind == "stack" else 0), 128, 3, "fp")
        elif c.kind == "trans":
            self.inp = nn.Linear(Z + extra, 128)
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.pol = mlp(128 + 4 + E.N_COLORS, 128, 3, "fp")
        self.ema = None

    # ---------------------------------------------------------------- helpers
    def make_ema(self):
        self.ema = copy.deepcopy(self.enc)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, decay=0.995):
        for pe, p in zip(self.ema.parameters(), self.enc.parameters()):
            pe.mul_(decay).add_(p.detach(), alpha=1 - decay)

    def init_state(self, n, device):
        return torch.zeros(n, H, device=device)

    def cell_step(self, h, z, valid, prop, task, a_prev):
        x = torch.cat([z * valid[:, None], valid[:, None], prop, task, a_prev], -1)
        return fq_state(self.cell(x, h), self.cfg.state_bits)

    def policy(self, h, prop, task, zin):
        """zin = z_t * valid_t (zeros on skipped/blacked-out frames): amendment 2, feed-through of the current embedding."""
        return self.pol(torch.cat([h, prop, task, zin], -1))

    # ---------------------------------------------------------------- sequence forward (training)
    def forward_seq(self, b, jepa_w=1.0):
        """b: dict of GPU tensors (B,T,...) with keys img u8, valid_in, valid_env, prop, task, a_exec, a_label, mask."""
        B, T = b["mask"].shape
        dev = b["mask"].device
        c = self.cfg
        a_prev = torch.cat([torch.zeros(B, 1, 3, device=dev), b["a_exec"][:, :-1]], 1)
        x = to_x(b["img"])
        tflat = b["task"].reshape(B * T, -1)
        valid = b["valid_in"].float()
        out = {}
        if c.kind == "stack":
            xs = x.view(B, T, 3, 32, 32) * valid[:, :, None, None, None]
            st, vs = [], []
            for k in range(4):
                st.append(F.pad(xs, (0, 0, 0, 0, 0, 0, k, 0))[:, :T])
                vs.append(F.pad(valid, (k, 0))[:, :T])
            xstack = torch.cat(st, 2).reshape(B * T, 12, 32, 32)
            z = self.enc(xstack, tflat).view(B, T, Z)
            feat = torch.cat([z, torch.stack(vs[1:], -1), valid[..., None], b["prop"], b["task"], a_prev], -1)
            logits = self.pol(feat)
        else:
            z = self.enc(x, tflat).view(B, T, Z)
            if c.kind == "react":
                feat = torch.cat([z * valid[..., None], valid[..., None], b["prop"], b["task"], a_prev], -1)
                logits = self.pol(feat)
            elif c.kind == "trans":
                tok = self.inp(torch.cat([z * valid[..., None], valid[..., None], b["prop"], b["task"], a_prev], -1))
                i = torch.arange(T, device=dev)
                allowed = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < 16)
                hh = self.tf(tok, mask=~allowed)
                logits = self.pol(torch.cat([hh, b["prop"], b["task"]], -1))
            else:
                h = self.init_state(B, dev)
                hs = []
                for t in range(T):
                    h = self.cell_step(h, z[:, t], valid[:, t], b["prop"][:, t], b["task"][:, t], a_prev[:, t])
                    hs.append(h)
                hs = torch.stack(hs, 1)
                logits = self.policy(hs, b["prop"], b["task"], z * valid[..., None])
                out["h"] = hs
        m = b["mask"].float()
        lab = b["a_label"]
        bc = (((torch.tanh(logits[..., :2]) - lab[..., :2]) ** 2).sum(-1)
              + 0.5 * F.binary_cross_entropy_with_logits(logits[..., 2], lab[..., 2], reduction="none"))
        out["bc"] = (bc * m).sum() / m.sum()
        loss = out["bc"]
        if c.kind == "rec" and c.jepa:
            with torch.no_grad():
                zt = F.normalize(self.ema(x, tflat).view(B, T, Z), dim=-1)
            zhat = self.pred(torch.cat([hs[:, :-1], b["a_exec"][:, :-1]], -1))
            zhat_n = F.normalize(zhat, dim=-1)
            tv = (b["valid_env"][:, 1:].float() * m[:, 1:])
            cos = (zhat_n * zt[:, 1:]).sum(-1)
            jl = ((1 - cos) * tv).sum() / tv.sum().clamp_min(1)
            out["jepa"] = jl
            loss = loss + jepa_w * jl
            with torch.no_grad():                         # trigger head regresses the (detached) prediction error
                surprise = (zhat_n - zt[:, 1:]).pow(2).sum(-1)
            pool = pool_feat(b["img"]).view(B, T, POOL_DIM)[:, 1:]
            u = self.trig(torch.cat([hs[:, :-1].detach(), b["a_exec"][:, :-1], pool], -1)).squeeze(-1)
            tl = (((u - surprise) ** 2) * tv).sum() / tv.sum().clamp_min(1)
            out["trig"] = tl
            loss = loss + 0.5 * tl
            out["z_std"] = z.detach()[b["mask"]].std(0).mean()
        out["loss"] = loss
        return out


def build(arm):
    n = Net(arm)
    if n.cfg.kind == "rec" and n.cfg.jepa:
        n.make_ema()
    return n


# ------------------------------------------------------------------------------------------ analytical cost model
def count_params(module, only=None):
    return sum(p.numel() for n, p in module.named_parameters() if only is None or only in n)
