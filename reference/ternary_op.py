"""
FlexTern golden reference for the ternary primitive.

This module is the software source of truth that the RTL in ``rtl/`` is
verified against. It is deliberately plain Python with no dependencies.

Semantics
---------
For an activation ``x`` and a ternary weight ``w`` in {-1, 0, +1}:

    w = +1  ->  +x
    w =  0  ->   0
    w = -1  ->  -x

``ternary_op`` operates on unbounded Python integers. Range limits (such as
the signed 8-bit activation range used by Y-Core v0.1) are the caller's
concern; the function itself is exact for any integer activation.

Run this file directly to execute its built-in self-test.
"""

from __future__ import annotations

from collections.abc import Sequence

VALID_WEIGHTS = (-1, 0, 1)


def ternary_op(x: int, w: int) -> int:
    """Return the product of activation ``x`` and ternary weight ``w``.

    Args:
        x: Integer activation. Any Python int is accepted.
        w: Ternary weight. Must be exactly -1, 0, or +1.

    Returns:
        ``+x`` if ``w == 1``, ``0`` if ``w == 0``, ``-x`` if ``w == -1``.

    Raises:
        ValueError: if ``w`` is not one of -1, 0, +1.
        TypeError: if ``x`` or ``w`` is not an int (bool is rejected too,
            since True/False would silently act as 1/0).
    """
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"activation must be int, got {type(x).__name__}")
    if isinstance(w, bool) or not isinstance(w, int):
        raise TypeError(f"weight must be int, got {type(w).__name__}")
    if w not in VALID_WEIGHTS:
        raise ValueError(f"ternary weight must be -1, 0, or +1; got {w}")
    if w == 1:
        return x
    if w == -1:
        return -x
    return 0


def ternary_dot(activations: Sequence[int], weights: Sequence[int]) -> int:
    """Return the ternary dot product ``sum(ternary_op(x_i, w_i))``.

    Args:
        activations: Sequence of integer activations.
        weights: Sequence of ternary weights, same length as ``activations``.

    Returns:
        The exact integer sum. No width limit is applied here.

    Raises:
        ValueError: if the two sequences differ in length, or if any weight
            is not a valid ternary value. Length is checked explicitly so a
            mismatch can never be silently truncated by ``zip``.
    """
    if len(activations) != len(weights):
        raise ValueError(
            f"length mismatch: {len(activations)} activations vs "
            f"{len(weights)} weights"
        )
    return sum(ternary_op(x, w) for x, w in zip(activations, weights))


def _self_test() -> None:
    """Minimal built-in check. The full suite is tests/test_reference.py."""
    assert ternary_op(12, 1) == 12
    assert ternary_op(12, 0) == 0
    assert ternary_op(12, -1) == -12

    assert ternary_op(-7, 1) == -7
    assert ternary_op(-7, 0) == 0
    assert ternary_op(-7, -1) == 7

    # The case that forces a 9-bit result in hardware: -128 * -1 = +128.
    assert ternary_op(-128, -1) == 128

    assert ternary_dot([5, 8, 3, 7], [1, -1, 0, 1]) == 4

    for bad in (2, -2, 5):
        try:
            ternary_op(1, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"weight {bad} should have been rejected")

    try:
        ternary_dot([1, 2], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch should have been rejected")

    print("PASS: ternary_op.py self-test")


if __name__ == "__main__":
    _self_test()
