"""
EXP-0005 Python golden reference for packed ternary matrix-vector products.

The packing interpretation is NOT re-derived here: pack/unpack are imported from
analysis/safetensors_reader.py, the routines transcribed from transformers'
integrations/bitnet.py and validated on all three canonical checkpoints in
EXP-0002/0002B (byte-level coverage, 2.08 B / 1.59 B / 2.92 B weights counted).

    packed shape [N/4, K] uint8;   packed[r, c] bits [2i+1:2i] = code of W[i*(N/4) + r, c]
    code = q + 1   (00 = -1, 01 = 0, 10 = +1, 11 unused)

Hardware semantics for the unused code 11 (deterministic, documented in
rtl/ternary_unpack4.sv): it contributes 0.  `decode_byte` implements exactly
that, and `test_hardware_gate.py` checks it agrees with `unpack_ternary` on
every valid code.

All arithmetic is exact Python integers; no numpy overflow is possible in the
golden results.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from safetensors_reader import pack_ternary, unpack_ternary  # noqa: E402

CANONICAL_K = (2048, 2560, 6912, 9216, 13312)    # EXP-0002/0002B, all three models
INT8_MIN, INT8_MAX = -128, 127


# --------------------------------------------------------------------------- codes
def decode_code(code: int) -> int:
    """Checkpoint 2-bit code -> ternary value with the hardware's reserved-code rule."""
    if code == 3:
        return 0            # reserved: contributes nothing (rtl/ternary_unpack4.sv)
    return code - 1


def decode_byte(b: int) -> tuple[int, int, int, int]:
    """Four ternary values of one packed byte, index i -> output row i*(N/4) + r."""
    return tuple(decode_code((b >> (2 * i)) & 3) for i in range(4))


def ycore_code(code: int) -> int:
    """Y-Core (ADR-0002) encoding of a checkpoint code: (code - 1) mod 4."""
    return (code - 1) & 3


def accumulator_bits(k: int, act_max_abs: int = 128) -> int:
    """Signed bits needed for sum of k INT8 x ternary terms: ceil(log2(k*act_max_abs + 1)) + 1."""
    m = k * act_max_abs
    return m.bit_length() + 1


def int4_accumulator_bits(k: int, act_max_abs: int = 128, w_max_abs: int = 8) -> int:
    m = k * act_max_abs * w_max_abs
    return m.bit_length() + 1


# --------------------------------------------------------------------------- golden
def matvec_from_packed(packed: np.ndarray, x: list[int] | np.ndarray) -> list[int]:
    """y[n] = sum_c x[c] * W[n, c] with W decoded from the packed bytes, exact ints.

    Reserved codes (11) contribute 0, matching the hardware.  N = 4 * rows.
    """
    rows, k = packed.shape
    assert len(x) == k, (len(x), k)
    xs = [int(v) for v in x]
    for v in xs:
        assert INT8_MIN <= v <= INT8_MAX
    n = 4 * rows
    y = [0] * n
    p = packed.astype(np.int64)
    for i in range(4):
        codes = (p >> (2 * i)) & 3                      # [rows, k]
        q = np.where(codes == 3, 0, codes - 1)         # reserved -> 0
        # exact: entries are small; int64 dot is exact well below 2^63 for k <= 13312
        part = q @ np.asarray(xs, dtype=np.int64)
        for r in range(rows):
            y[i * rows + r] = int(part[r])
    return y


def matvec_dense(q: np.ndarray, x) -> list[int]:
    """Reference on an unpacked int8 ternary matrix [N, K] (cross-check path)."""
    return [int(v) for v in (q.astype(np.int64) @ np.asarray([int(v) for v in x], dtype=np.int64))]


def rowq_to_row(rowq: int, i: int, n: int) -> int:
    """Output row produced by accumulator i of row-quad rowq (N outputs)."""
    return i * (n // 4) + rowq


# --------------------------------------------------------------------------- INT4 baseline packing
def pack_int4(w: np.ndarray) -> np.ndarray:
    """Signed INT4 weights [N, K] -> [N/2, K] bytes, two output rows per byte along N.

    byte[r, c] low nibble = W[r, c], high nibble = W[r + N/2, c], two's complement.
    Mirrors the ternary packing's 'output rows per byte at one K position'.
    """
    n, k = w.shape
    assert n % 2 == 0
    half = n // 2
    lo = (w[:half].astype(np.int16) & 0xF).astype(np.uint8)
    hi = (w[half:].astype(np.int16) & 0xF).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


def int4_matvec_from_packed(packed: np.ndarray, x) -> list[int]:
    rows, k = packed.shape
    p = packed.astype(np.int64)
    xs = np.asarray([int(v) for v in x], dtype=np.int64)
    y = [0] * (2 * rows)
    for i in range(2):
        nib = (p >> (4 * i)) & 0xF
        w = np.where(nib >= 8, nib - 16, nib)
        part = w @ xs
        for r in range(rows):
            y[i * rows + r] = int(part[r])
    return y


# --------------------------------------------------------------------------- vector files
def to_twos(v: int, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not lo <= v <= hi:
        raise ValueError(f"{v} does not fit in {bits} signed bits")
    return v & ((1 << bits) - 1)


def write_case(prefix: str, packed: np.ndarray, xs: list[list[int]], acc_w: int, ref_fn=matvec_from_packed,
               rows_per_byte: int = 4, reserved: bool = False) -> dict:
    """Write <prefix>_w.hex (one packed byte per line, row-major), <prefix>_x{m}.hex, <prefix>_y{m}.hex
    (acc_w-bit two's complement per output row), and <prefix>_cfg.txt (K N n_ops rows acc_w reserved_expected).
    Returns a provenance dict with the SHA-256 of the packed bytes."""
    rows, k = packed.shape
    n = rows_per_byte * rows
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    with open(prefix + "_w.hex", "w") as f:
        f.write("\n".join(f"{int(b):02x}" for b in packed.reshape(-1)) + "\n")
    for m, x in enumerate(xs):
        y = ref_fn(packed, x)
        with open(f"{prefix}_x{m}.hex", "w") as f:
            f.write("\n".join(f"{to_twos(int(v), 8):02x}" for v in x) + "\n")
        with open(f"{prefix}_y{m}.hex", "w") as f:
            f.write("\n".join(f"{to_twos(int(v), acc_w):0{(acc_w + 3) // 4}x}" for v in y) + "\n")
    with open(prefix + "_cfg.txt", "w") as f:
        f.write(f"{k} {n} {len(xs)} {rows} {acc_w} {int(reserved)}\n")
    return {"k": k, "n": n, "rows": rows, "ops": len(xs), "packed_bytes": int(packed.nbytes),
            "acc_w": acc_w, "reserved": bool(reserved), "sha256_packed": hashlib.sha256(packed.tobytes()).hexdigest()}


def random_ternary_packed(rng: np.random.Generator, n: int, k: int, p_zero: float = 0.4) -> np.ndarray:
    q = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, k), p=[(1 - p_zero) / 2, p_zero, (1 - p_zero) / 2])
    return pack_ternary(q)


def random_activations(rng: np.random.Generator, k: int, lo: int = -127, hi: int = 127) -> list[int]:
    return [int(v) for v in rng.integers(lo, hi + 1, size=k)]


def roundtrip_check(packed: np.ndarray) -> bool:
    """unpack -> pack reproduces the bytes exactly (valid codes only)."""
    return bool(np.array_equal(pack_ternary(unpack_ternary(packed)), packed))
