"""
Validation tests for the EXP-0002 workload-characterization tooling.

Run from the repository root:
    python -m unittest tests.test_analysis -v

No model download is needed. Tests use synthetic tensors and a synthetic
safetensors file written to a temp directory.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "analysis"))

from safetensors_reader import SafetensorsFile, bf16_to_f32, pack_ternary, unpack_ternary  # noqa: E402
from ternary_model_profile import accumulator_width, brute_force_accumulator, signed_bits_for_range  # noqa: E402


class TestUnpack(unittest.TestCase):
    def test_transformers_docstring_example(self):
        # Verbatim from transformers/integrations/bitnet.py unpack_weights docstring.
        packed = np.array([[0b10100001, 0b00011000],
                           [0b10010000, 0b00001010]], dtype=np.uint8)
        expected = np.array([[0, -1],
                             [-1, 1],
                             [-1, 1],
                             [-1, 1],
                             [1, 0],
                             [0, -1],
                             [1, -1],
                             [1, -1]], dtype=np.int8)
        np.testing.assert_array_equal(unpack_ternary(packed), expected)

    def test_round_trip(self):
        rng = np.random.default_rng(0)  # test-only seed; the analysis itself is deterministic
        q = rng.integers(-1, 2, size=(64, 37), dtype=np.int8)
        np.testing.assert_array_equal(unpack_ternary(pack_ternary(q)), q)

    def test_exact_counting(self):
        q = np.array([[-1, 0, 1, 1], [0, 0, 0, -1], [1, 1, 1, 1], [-1, -1, 0, 0]], dtype=np.int8)
        u = unpack_ternary(pack_ternary(q))
        # rows: [-1,0,1,1] [0,0,0,-1] [1,1,1,1] [-1,-1,0,0] -> four -1, six 0, six +1
        self.assertEqual(int((u == -1).sum()), 4)
        self.assertEqual(int((u == 0).sum()), 6)
        self.assertEqual(int((u == 1).sum()), 6)
        self.assertEqual(4 + 6 + 6, u.size)

    def test_code3_is_not_folded(self):
        packed = np.array([[0b11111111]], dtype=np.uint8)  # four code-3 values
        u = unpack_ternary(packed)
        self.assertTrue((u == 2).all(), "reserved code 3 must surface as +2, not be silently mapped")

    def test_rejects_non_uint8(self):
        with self.assertRaises(TypeError):
            unpack_ternary(np.zeros((4, 4), dtype=np.int8))


class TestBF16(unittest.TestCase):
    def test_known_values(self):
        # bf16 0x3F80 = 1.0, 0xBF80 = -1.0, 0x3C00 = 0.0078125 (2^-7), 0x0000 = 0.0
        u16 = np.array([0x3F80, 0xBF80, 0x3C00, 0x0000], dtype=np.uint16)
        np.testing.assert_array_equal(bf16_to_f32(u16), np.array([1.0, -1.0, 0.0078125, 0.0], dtype=np.float32))


class TestSafetensorsReader(unittest.TestCase):
    def _write(self, tensors: dict[str, tuple[str, np.ndarray]]) -> str:
        """tensors: name -> (dtype_str, raw bytes array). Writes a valid safetensors file."""
        header = {"__metadata__": {"format": "pt"}}
        blobs = []
        off = 0
        for name, (dt, arr) in tensors.items():
            b = arr.tobytes()
            header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
        hj = json.dumps(header).encode("utf-8")
        path = os.path.join(self.tmp, "m.safetensors")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            for b in blobs:
                f.write(b)
        return path

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_shapes_bytes_and_layout(self):
        q = np.array([[-1, 0, 1, 1], [0, 0, 0, -1], [1, 1, 1, 1], [-1, -1, 0, 0],
                      [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int8)  # 8x4
        packed = pack_ternary(q)  # 2x4 uint8
        scale = np.array([0x3C00], dtype="<u2")  # bf16 2^-7
        path = self._write({"w.weight": ("U8", packed), "w.weight_scale": ("BF16", scale)})
        st = SafetensorsFile(path)
        layout = st.verify_layout()
        self.assertEqual(layout["tensor_count"], 2)
        self.assertEqual(st.tensors["w.weight"].shape, (2, 4))
        self.assertEqual(st.tensors["w.weight"].nbytes, 8)
        np.testing.assert_array_equal(unpack_ternary(st.uint8("w.weight")), q)
        self.assertAlmostEqual(float(st.bf16_as_f32("w.weight_scale")[0]), 0.0078125)
        self.assertEqual(layout["file_size"], os.path.getsize(path))

    def test_layout_detects_truncation(self):
        packed = np.zeros((2, 4), dtype=np.uint8)
        path = self._write({"w": ("U8", packed)})
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 1)
        st = SafetensorsFile(path)
        with self.assertRaises(AssertionError):
            st.verify_layout()


class TestAccumulatorWidth(unittest.TestCase):
    def test_signed_bits(self):
        self.assertEqual(signed_bits_for_range(-128, 127), 8)
        self.assertEqual(signed_bits_for_range(-128, 128), 9)
        self.assertEqual(signed_bits_for_range(-2048, 2048), 13)  # Phase 1 Y-Dot16
        self.assertEqual(signed_bits_for_range(-2048, 2047), 12)

    def test_phase1_reference(self):
        r = accumulator_width(16, -128, 127)
        self.assertEqual((r["sum_min"], r["sum_max"], r["signed_bits"]), (-2048, 2048, 13))
        r = accumulator_width(1, -128, 127)
        self.assertEqual((r["sum_min"], r["sum_max"], r["signed_bits"]), (-128, 128, 9))

    def test_brute_force_k1_k2_format_range(self):
        for k in (1, 2):
            lo, hi = brute_force_accumulator(k, -128, 127)
            r = accumulator_width(k, -128, 127)
            self.assertEqual((lo, hi), (r["sum_min"], r["sum_max"]), f"K={k}")

    def test_brute_force_k1_k2_k3_model_range(self):
        # Model range is symmetric; smaller domain so K=3 is still cheap (255^3 * 27 is too big; use subset range)
        for k in (1, 2):
            lo, hi = brute_force_accumulator(k, -127, 127)
            r = accumulator_width(k, -127, 127)
            self.assertEqual((lo, hi), (r["sum_min"], r["sum_max"]))
        lo, hi = brute_force_accumulator(3, -5, 4)
        r = accumulator_width(3, -5, 4)
        self.assertEqual((lo, hi), (r["sum_min"], r["sum_max"]))

    def test_model_k_values(self):
        # K = 2560: 128*2560 = 327,680 -> needs 2^(b-1) >= 327,681 -> b = 20
        self.assertEqual(accumulator_width(2560, -128, 127)["signed_bits"], 20)
        # K = 6912: 128*6912 = 884,736 -> 2^20 = 1,048,576 >= 884,737 -> b = 21
        self.assertEqual(accumulator_width(6912, -128, 127)["signed_bits"], 21)
        # Model range at 2560: 127*2560 = 325,120 -> still 20 bits
        self.assertEqual(accumulator_width(2560, -127, 127)["signed_bits"], 20)


class TestActivationQuantReproduction(unittest.TestCase):
    """Reproduce transformers activation_quant on synthetic inputs and check -128 is unreachable."""

    @staticmethod
    def act_quant(x: np.ndarray):
        scale = 127.0 / np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
        return np.clip(np.round(x * scale), -128, 127).astype(np.int8), scale

    def test_reference_examples(self):
        q, _ = self.act_quant(np.array([[1.0, -2.0, 0.5, 0.0]]))
        np.testing.assert_array_equal(q, np.array([[64, -127, 32, 0]], dtype=np.int8))  # 63.5 rounds to 64 (banker's) ; 31.75 -> 32

    def test_minus_128_unreachable(self):
        rng = np.random.default_rng(1)  # test-only
        x = rng.standard_normal((2000, 64)) * rng.uniform(0.001, 100, (2000, 1))
        x[:, 0] = -np.abs(x).max(axis=1) * 1.0  # force the absmax element to be negative
        q, _ = self.act_quant(x)
        self.assertEqual(int(q.min()), -127)
        self.assertEqual(int(q.max()), 127)

    def test_weight_ternarization_reproduction(self):
        # WeightQuant.forward: scale = 1/mean|W|; q = round(W*scale).clamp(-1,1)
        w = np.array([[0.30, -0.01, 0.05, -0.20], [0.0, 0.12, -0.33, 0.02]], dtype=np.float32)
        scale = 1.0 / max(np.abs(w).mean(), 1e-5)
        q = np.clip(np.round(w * scale), -1, 1)
        # mean|W| = 1.03/8 = 0.12875 -> scale ~7.767; 0.05*7.767=0.388->0; 0.12*7.767=0.93->1; 0.02*7.767=0.155->0
        np.testing.assert_array_equal(q, np.array([[1, 0, 0, -1], [0, 1, -1, 0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
