"""
Generate Y-Dot16 cross-check vectors from the Python golden reference.

Writes a $readmemh-compatible file in which each line is one 176-bit word
(44 hex digits):

    { 3'b000, expected[12:0], w_flat[31:0], x_flat[127:0] }

  x_flat[8*i +: 8]  = activation i as 8-bit two's complement
  w_flat[2*i +: 2]  = weight i, encoded 00=0, 01=+1, 11=-1
  expected[12:0]    = ternary_dot(x, w) as 13-bit two's complement

The RTL testbench tests/y_dot16_tb.sv loads this file and compares the DUT
output to the expected column, so the RTL is checked against an
independently written implementation (reference/ternary_op.py), not against
arithmetic written in the testbench itself.

Usage (from repo root):
    python scripts/gen_vectors.py [--count 2000] [--seed 20261005] [--out tests/vectors/y_dot16_vectors.hex]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "reference"))

from ternary_op import ternary_dot  # noqa: E402

N = 16
WEIGHT_ENC = {0: 0b00, 1: 0b01, -1: 0b11}


def to_twos(value: int, bits: int) -> int:
    """Two's-complement encode ``value`` in ``bits`` bits, with range check."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not lo <= value <= hi:
        raise ValueError(f"{value} does not fit in {bits} bits signed")
    return value & ((1 << bits) - 1)


def pack_vector(xs: list[int], ws: list[int]) -> int:
    x_flat = 0
    w_flat = 0
    for i, (x, w) in enumerate(zip(xs, ws)):
        x_flat |= to_twos(x, 8) << (8 * i)
        w_flat |= WEIGHT_ENC[w] << (2 * i)
    expected = to_twos(ternary_dot(xs, ws), 13)
    return (expected << 160) | (w_flat << 128) | x_flat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default=os.path.join("tests", "vectors", "y_dot16_vectors.hex"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lines = []
    # Always include the worst cases first so they are never left to chance.
    fixed = [
        ([127] * N, [1] * N),
        ([-128] * N, [1] * N),
        ([-128] * N, [-1] * N),
        ([127] * N, [-1] * N),
        ([5, 8, 3, 7] + [0] * 12, [1, -1, 0, 1] + [0] * 12),
    ]
    for xs, ws in fixed:
        lines.append(f"{pack_vector(xs, ws):044x}")
    while len(lines) < args.count:
        xs = [rng.randint(-128, 127) for _ in range(N)]
        ws = [rng.choice((-1, 0, 1)) for _ in range(N)]
        lines.append(f"{pack_vector(xs, ws):044x}")

    with open(args.out, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} vectors to {args.out} (seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
