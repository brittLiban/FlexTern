"""
Cross-check the official absmean ternarization against a packed checkpoint.

For the model's spec (analysis/model_specs.py), fetches a few BF16 latent
tensors from the latent repo/revision by HTTP range (tens of MB, no full
download), applies absmean ternarization (scale = 1/mean|W|;
q = round(W*scale).clamp(-1,1); torch semantics via numpy round-half-even),
and compares with the unpacked 2-bit tensor from the packed checkpoint.
Also checks the stored weight_scale against mean|W| under the spec's
declared scale convention:
    multiply: stored == mean|W|
    divide:   stored == 1/mean|W|   (so stored * mean|W| == 1)

Writes <default_out>/ternarization_crosscheck.json.

Usage (from repo root, after ternary_model_profile.py has cached the packed file):
    python analysis/ternarization_crosscheck.py [--model bitnet-2b4t|falcon-e-1b]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_specs import SPECS, get_spec  # noqa: E402
from safetensors_reader import SafetensorsFile, bf16_to_f32, unpack_ternary  # noqa: E402

HDR = {"User-Agent": "flextern-analysis"}


def http_range(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={**HDR, "Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req).read()


def ternarize_absmean(w: np.ndarray) -> np.ndarray:
    """absmean ternarization with torch .round() semantics (half-to-even), float32 math."""
    scale = 1.0 / max(float(np.abs(w).mean()), 1e-5)
    return np.clip(np.round(w * scale), -1, 1).astype(np.int8)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest BF16 (round-to-nearest-even) and return as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def ternarize_absmean_bf16(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Same, but emulating a model loaded in torch.bfloat16: mean, reciprocal, and the product
    w*scale are each rounded to BF16 before torch.round(). This is what onebitllms.quantize_to_1bit
    does (from_pretrained(torch_dtype=bfloat16)). Returns (q, scale_as_stored)."""
    beta = float(to_bf16(np.array([np.abs(w).mean()], dtype=np.float32))[0])
    scale = float(to_bf16(np.array([1.0 / max(beta, 1e-5)], dtype=np.float32))[0])
    return np.clip(np.round(to_bf16(w * np.float32(scale))), -1, 1).astype(np.int8), scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bitnet-2b4t", choices=sorted(SPECS))
    ap.add_argument("--out", help="output directory (default: the model's profile directory)")
    args = ap.parse_args()
    spec = get_spec(args.model)
    if not spec.latent_repo or not spec.crosscheck_tensors:
        raise SystemExit(f"{spec.slug}: no latent repo / cross-check tensors declared")
    out = args.out or spec.default_out

    from huggingface_hub import hf_hub_download
    packed_path = hf_hub_download(spec.model_id, spec.weights_file, revision=spec.revision)
    st = SafetensorsFile(packed_path)

    url = f"https://huggingface.co/{spec.latent_repo}/resolve/{spec.latent_revision}/model.safetensors"
    n = struct.unpack("<Q", http_range(url, 0, 7))[0]
    hdr = json.loads(http_range(url, 8, 8 + n - 1))
    results = []
    for name in spec.crosscheck_tensors:
        t = hdr[name]
        o0, o1 = t["data_offsets"]
        raw = http_range(url, 8 + n + o0, 8 + n + o1 - 1)
        w = bf16_to_f32(np.frombuffer(raw, dtype="<u2")).reshape(t["shape"])
        beta = float(np.abs(w).mean())
        q_ref = ternarize_absmean(w)
        q_pk = unpack_ternary(st.uint8(name))
        ws = float(st.bf16_as_f32(name + "_scale").reshape(-1)[0])
        if spec.scale_convention == "multiply":
            expected_scale, conv_ok = beta, abs(ws - beta) / beta < 0.01
        else:
            expected_scale, conv_ok = 1.0 / beta, abs(ws * beta - 1.0) < 0.01
        agree = float((q_ref == q_pk).mean())
        q_emu, _ = ternarize_absmean_bf16(w)
        agree_emu = float((q_emu == q_pk).mean())
        r = {"tensor": name, "shape": t["shape"], "bf16_mean_abs": beta, "packed_weight_scale": ws,
             "scale_convention": spec.scale_convention, "expected_stored_scale": expected_scale,
             "stored_scale_times_mean_abs": ws * beta, "scale_convention_confirmed_to_bf16_precision": conv_ok,
             "agreement_pct_float32_math": 100 * agree, "mismatches_float32_math": int((q_ref != q_pk).sum()),
             "agreement_pct_bf16_emulated": 100 * agree_emu, "mismatches_bf16_emulated": int((q_emu != q_pk).sum()),
             "elements": int(q_pk.size),
             "zero_pct_from_latent": 100 * float((q_ref == 0).mean()), "zero_pct_packed": 100 * float((q_pk == 0).mean())}
        results.append(r)
        print(f"{name}: mean|W|={beta:.5f} stored_scale={ws:.5f} (expected {expected_scale:.5f}, {spec.scale_convention}, ok={conv_ok}) "
              f"agree f32={100*agree:.3f}% bf16-emulated={100*agree_emu:.3f}%  zero% latent->{r['zero_pct_from_latent']:.2f} packed {r['zero_pct_packed']:.2f}")
    outj = {"packed_repo": spec.model_id, "packed_revision": spec.revision,
            "latent_repo": spec.latent_repo, "latent_revision": spec.latent_revision, "latent_note": spec.latent_note,
            "method": "absmean ternarization: scale=1/mean|W|, q=round(W*scale).clamp(-1,1), round-half-even; reported twice: "
                      "float32 math, and with BF16 emulation (mean, reciprocal and product each rounded to BF16, as a model "
                      "loaded in torch.bfloat16 would compute)",
            "scale_convention": spec.scale_convention, "tensors": results}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ternarization_crosscheck.json"), "w", encoding="utf-8") as f:
        json.dump(outj, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
