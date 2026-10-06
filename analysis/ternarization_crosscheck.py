"""
Cross-check the official absmean ternarization against the packed checkpoint.

Fetches a few BF16 latent tensors from the sibling repo
microsoft/bitnet-b1.58-2B-4T-bf16 by HTTP range (tens of MB, no full
download), applies WeightQuant.forward from transformers/integrations/bitnet.py
(scale = 1/mean|W|; q = round(W*scale).clamp(-1,1)), and compares with the
unpacked 2-bit tensor from the packed repo. Also compares mean|W| with the
packed repo's stored weight_scale.

Writes results/workload_profile/ternarization_crosscheck.json.

Usage (from repo root, after ternary_model_profile.py has cached the packed file):
    python analysis/ternarization_crosscheck.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import SafetensorsFile, bf16_to_f32, unpack_ternary  # noqa: E402
from ternary_model_profile import MODEL_ID, MODEL_REVISION  # noqa: E402

BF16_REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
BF16_REVISION = "276681394656abdadb8e80e5b2c3db5e5d7fcaff"
TENSORS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.17.mlp.down_proj.weight",
]
HDR = {"User-Agent": "flextern-analysis"}


def http_range(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={**HDR, "Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req).read()


def ternarize_absmean(w: np.ndarray) -> np.ndarray:
    """WeightQuant.forward semantics (float32 math)."""
    scale = 1.0 / max(float(np.abs(w).mean()), 1e-5)
    return np.clip(np.round(w * scale), -1, 1).astype(np.int8)


def main() -> int:
    from huggingface_hub import hf_hub_download
    packed_path = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    st = SafetensorsFile(packed_path)

    url = f"https://huggingface.co/{BF16_REPO}/resolve/{BF16_REVISION}/model.safetensors"
    n = struct.unpack("<Q", http_range(url, 0, 7))[0]
    hdr = json.loads(http_range(url, 8, 8 + n - 1))
    results = []
    for name in TENSORS:
        t = hdr[name]
        o0, o1 = t["data_offsets"]
        raw = http_range(url, 8 + n + o0, 8 + n + o1 - 1)
        w = bf16_to_f32(np.frombuffer(raw, dtype="<u2")).reshape(t["shape"])
        beta = float(np.abs(w).mean())
        q_ref = ternarize_absmean(w)
        q_pk = unpack_ternary(st.uint8(name))
        ws = float(st.bf16_as_f32(name + "_scale").reshape(-1)[0])
        agree = float((q_ref == q_pk).mean())
        r = {"tensor": name, "shape": t["shape"], "bf16_mean_abs": beta, "packed_weight_scale": ws,
             "scale_matches_absmean_to_bf16_precision": abs(ws - beta) / beta < 0.01,
             "agreement_pct": 100 * agree, "mismatches": int((q_ref != q_pk).sum()), "elements": int(q_pk.size),
             "zero_pct_from_bf16": 100 * float((q_ref == 0).mean()), "zero_pct_packed": 100 * float((q_pk == 0).mean())}
        results.append(r)
        print(f"{name}: mean|W|={beta:.5f} weight_scale={ws:.5f} agree={100*agree:.3f}%  zero% bf16->{r['zero_pct_from_bf16']:.2f} packed {r['zero_pct_packed']:.2f}")
    out = {"packed_repo": MODEL_ID, "packed_revision": MODEL_REVISION, "bf16_repo": BF16_REPO, "bf16_revision": BF16_REVISION,
           "method": "WeightQuant.forward (transformers integrations/bitnet.py): scale=1/mean|W|, q=round(W*scale).clamp(-1,1), float32 math",
           "interpretation": "weight_scale stored in the packed repo equals mean|W_latent| (absmean beta) to BF16 precision; "
                             "ternarizing the bf16 repo reproduces the packed codes at ~98-99.4%, not 100%, so the two repos are not "
                             "bit-identical snapshots. The packed repo is the deployed artifact and is what EXP-0002 measures.",
           "tensors": results}
    os.makedirs(os.path.join("results", "workload_profile"), exist_ok=True)
    with open(os.path.join("results", "workload_profile", "ternarization_crosscheck.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
