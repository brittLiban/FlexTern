"""
EXP-0005 test-vector generator.

Writes case files (see reference.write_case) into an output directory:

  synthetic cases (seeded, reproducible from a clean checkout, never committed):
    rand_k<K>      random ternary matrix (40% zeros), N = 64 or 32, two activation vectors
    ext_k<K>       extremes: rows all +1 / all -1 / all 0 / alternating; activations
                   +127, -128 (format maximum), 0, -127  -> accumulator extremes per K
    reserved_k2048 valid matrix with reserved code 11 injected in 5% of bytes (expected: 0)
    tail_k2048     N/4 = 13 row-quads (not a multiple of any engine width or bank count)
    int4_k<K>      INT4 baseline cases (2 rows per byte, 25-bit accumulator)

  checkpoint slices (only when --checkpoints and the pinned files are in the local
  HF cache; provenance recorded, bytes written to the build directory only):
    ckpt_f1b_k_proj     Falcon-E 1B layer 0 k_proj, full packed tensor [64, 2048]  (N = 256)
    ckpt_f1b_down       Falcon-E 1B layer 0 down_proj, first 8 packed rows [8, 9216]
    ckpt_f3b_down       Falcon-E 3B layer 0 down_proj, first 8 packed rows [8, 13312]
    ckpt_bn_q_proj      BitNet 2B4T layer 0 q_proj, first 8 packed rows [8, 2560]
    ckpt_bn_down        BitNet 2B4T layer 0 down_proj, first 8 packed rows [8, 6912]

Usage: python analysis/hardware_gate/vectors.py --out build/vectors [--checkpoints] [--seed 20261007]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from reference import (CANONICAL_K, int4_matvec_from_packed, pack_int4, random_activations,  # noqa: E402
                       random_ternary_packed, write_case)
from safetensors_reader import SafetensorsFile, pack_ternary  # noqa: E402
from model_specs import SPECS  # noqa: E402

ACC_W = 22
INT4_ACC_W = 25

CKPT_SLICES = [
    ("ckpt_f1b_k_proj", "falcon-e-1b", "model.layers.0.self_attn.k_proj.weight", None),
    ("ckpt_f1b_down",   "falcon-e-1b", "model.layers.0.mlp.down_proj.weight", 8),
    ("ckpt_f3b_down",   "falcon-e-3b", "model.layers.0.mlp.down_proj.weight", 8),
    ("ckpt_bn_q_proj",  "bitnet-2b4t", "model.layers.0.self_attn.q_proj.weight", 8),
    ("ckpt_bn_down",    "bitnet-2b4t", "model.layers.0.mlp.down_proj.weight", 8),
]


def local_checkpoint_path(slug: str) -> str | None:
    """Path of the pinned safetensors file in the local HF cache, or None (no network)."""
    spec = SPECS[slug]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    repo = "models--" + spec.model_id.replace("/", "--")
    p = os.path.join(home, "hub", repo, "snapshots", spec.revision, spec.weights_file)
    return p if os.path.isfile(p) else None


def synthetic_cases(out: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cases = {}
    for k in CANONICAL_K:
        n = 64 if k <= 2560 else 32
        packed = random_ternary_packed(rng, n, k)
        xs = [random_activations(rng, k), random_activations(rng, k, -128, 127)]
        cases[f"rand_k{k}"] = write_case(os.path.join(out, f"rand_k{k}"), packed, xs, ACC_W) | {"kind": "synthetic"}
        # extremes: 8 rows = 2 row-quads
        q = np.zeros((8, k), dtype=np.int8)
        q[0, :] = 1; q[1, :] = -1; q[2, :] = 0; q[3, ::2] = 1; q[3, 1::2] = -1
        q[4, :] = 1; q[5, :] = -1; q[6, :] = 1; q[7, :] = -1
        packed = pack_ternary(q)
        xs = [[127] * k, [-128] * k, [0] * k, [-127] * k]
        cases[f"ext_k{k}"] = write_case(os.path.join(out, f"ext_k{k}"), packed, xs, ACC_W) | {"kind": "synthetic-extremes"}
    # reserved code 11 injected; expected contribution 0
    k = 2048
    packed = random_ternary_packed(rng, 64, k).copy()
    mask = rng.random(packed.shape) < 0.05
    inj = rng.integers(0, 4, size=packed.shape).astype(np.uint8)
    packed[mask] = packed[mask] | (np.uint8(3) << (2 * inj[mask]))
    assert int(mask.sum()) > 0
    cases["reserved_k2048"] = write_case(os.path.join(out, "reserved_k2048"), packed, [random_activations(rng, k)], ACC_W,
                                         reserved=True) | {"kind": "synthetic-reserved"}
    # tail: 13 row-quads
    packed = random_ternary_packed(rng, 52, k)
    cases["tail_k2048"] = write_case(os.path.join(out, "tail_k2048"), packed, [random_activations(rng, k), random_activations(rng, k)], ACC_W) | {"kind": "synthetic-tail"}
    # INT4 baseline cases
    for k in (2048, 13312):
        w = rng.integers(-8, 8, size=(32, k)).astype(np.int8)
        packed = pack_int4(w)
        xs = [random_activations(rng, k), [127] * k]
        cases[f"int4_k{k}"] = write_case(os.path.join(out, f"int4_k{k}"), packed, xs, INT4_ACC_W, ref_fn=int4_matvec_from_packed,
                                         rows_per_byte=2) | {"kind": "int4-synthetic"}
    return cases


def checkpoint_cases(out: str, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    cases = {}
    for name, slug, tensor, nrows in CKPT_SLICES:
        path = local_checkpoint_path(slug)
        if path is None:
            cases[name] = {"kind": "checkpoint", "skipped": f"{slug} not in local HF cache"}
            continue
        st = SafetensorsFile(path)
        full = st.uint8(tensor)
        packed = np.ascontiguousarray(full if nrows is None else full[:nrows])
        k = packed.shape[1]
        xs = [random_activations(rng, k), random_activations(rng, k)]
        prov = write_case(os.path.join(out, name), packed, xs, ACC_W)
        cases[name] = prov | {"kind": "checkpoint", "model_id": SPECS[slug].model_id, "revision": SPECS[slug].revision,
                              "tensor": tensor, "full_packed_shape": list(full.shape), "slice_rows": packed.shape[0],
                              "file_sha256_expected": SPECS[slug].expected_sha256}
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("build", "vectors"))
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--checkpoints", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = synthetic_cases(a.out, a.seed)
    if a.checkpoints:
        cases.update(checkpoint_cases(a.out, a.seed))
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "cases": cases}, f, indent=1)
    print(f"{len(cases)} cases written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
