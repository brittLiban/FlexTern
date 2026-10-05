"""
Unit tests for the FlexTern Python golden reference (reference/ternary_op.py).

Run from the repository root:

    python -m unittest tests.test_reference -v

Uses only the standard library.
"""

from __future__ import annotations

import os
import sys
import unittest

# Make reference/ importable without packaging.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "reference"))

from ternary_op import ternary_dot, ternary_op  # noqa: E402

INT8_MIN = -128
INT8_MAX = 127


class TestTernaryOp(unittest.TestCase):
    def test_positive_activation(self):
        self.assertEqual(ternary_op(12, 1), 12)
        self.assertEqual(ternary_op(12, 0), 0)
        self.assertEqual(ternary_op(12, -1), -12)

    def test_negative_activation(self):
        self.assertEqual(ternary_op(-7, 1), -7)
        self.assertEqual(ternary_op(-7, 0), 0)
        self.assertEqual(ternary_op(-7, -1), 7)

    def test_zero_activation(self):
        for w in (-1, 0, 1):
            self.assertEqual(ternary_op(0, w), 0)

    def test_plus_one_weight_is_identity(self):
        for x in range(INT8_MIN, INT8_MAX + 1):
            self.assertEqual(ternary_op(x, 1), x)

    def test_zero_weight_is_zero(self):
        for x in range(INT8_MIN, INT8_MAX + 1):
            self.assertEqual(ternary_op(x, 0), 0)

    def test_minus_one_weight_is_negation(self):
        for x in range(INT8_MIN, INT8_MAX + 1):
            self.assertEqual(ternary_op(x, -1), -x)

    def test_boundary_values(self):
        self.assertEqual(ternary_op(INT8_MIN, 1), -128)
        self.assertEqual(ternary_op(INT8_MIN, -1), 128)  # exceeds int8; needs 9 bits
        self.assertEqual(ternary_op(INT8_MAX, 1), 127)
        self.assertEqual(ternary_op(INT8_MAX, -1), -127)
        self.assertEqual(ternary_op(INT8_MIN, 0), 0)
        self.assertEqual(ternary_op(INT8_MAX, 0), 0)

    def test_exhaustive_int8_domain_matches_multiplication(self):
        # 256 activations x 3 weights = 768 cases, same domain the RTL TB covers.
        count = 0
        for x in range(INT8_MIN, INT8_MAX + 1):
            for w in (-1, 0, 1):
                self.assertEqual(ternary_op(x, w), x * w)
                count += 1
        self.assertEqual(count, 768)

    def test_invalid_weights_raise(self):
        for bad in (2, -2, 3, 100, -100):
            with self.assertRaises(ValueError):
                ternary_op(1, bad)

    def test_non_int_inputs_raise(self):
        with self.assertRaises(TypeError):
            ternary_op(1.0, 1)
        with self.assertRaises(TypeError):
            ternary_op(1, 1.0)
        with self.assertRaises(TypeError):
            ternary_op(True, 1)
        with self.assertRaises(TypeError):
            ternary_op(1, True)


class TestTernaryDot(unittest.TestCase):
    def test_spec_example(self):
        self.assertEqual(ternary_dot([5, 8, 3, 7], [1, -1, 0, 1]), 4)

    def test_empty(self):
        self.assertEqual(ternary_dot([], []), 0)

    def test_all_zero_weights(self):
        self.assertEqual(ternary_dot([1, 2, 3, 4], [0, 0, 0, 0]), 0)

    def test_all_plus_one(self):
        self.assertEqual(ternary_dot([1, 2, 3, 4], [1, 1, 1, 1]), 10)

    def test_all_minus_one(self):
        self.assertEqual(ternary_dot([1, 2, 3, 4], [-1, -1, -1, -1]), -10)

    def test_16_element_boundaries(self):
        # These are the Y-Dot16 worst cases: |result| up to 2048, needs 13 bits.
        self.assertEqual(ternary_dot([127] * 16, [1] * 16), 2032)
        self.assertEqual(ternary_dot([-128] * 16, [1] * 16), -2048)
        self.assertEqual(ternary_dot([-128] * 16, [-1] * 16), 2048)
        self.assertEqual(ternary_dot([127] * 16, [-1] * 16), -2032)

    def test_matches_naive_sum(self):
        import random

        rng = random.Random(1234)
        for _ in range(2000):
            xs = [rng.randint(INT8_MIN, INT8_MAX) for _ in range(16)]
            ws = [rng.choice((-1, 0, 1)) for _ in range(16)]
            self.assertEqual(ternary_dot(xs, ws), sum(x * w for x, w in zip(xs, ws)))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ternary_dot([1, 2, 3], [1, 1])
        with self.assertRaises(ValueError):
            ternary_dot([1, 2], [1, 1, 1])

    def test_invalid_weight_in_vector_raises(self):
        with self.assertRaises(ValueError):
            ternary_dot([1, 2, 3], [1, 2, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
