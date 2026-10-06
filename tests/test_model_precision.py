"""
Validation tests for EXP-0004 tooling.

Run from the repository root:
    python -m unittest tests.test_model_precision -v

CPU-only tests always run. Tests that need torch+CUDA, the model, or optional packages skip
cleanly when those are absent. Set FLEXTERN_SLOW_TESTS=1 to enable the model-level checks.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "model_precision"))
os.chdir(_ROOT)

try:
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

SLOW = os.environ.get("FLEXTERN_SLOW_TESTS") == "1"


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestHeadRTN(unittest.TestCase):
    def setUp(self):
        from head_quantization import quantize_rtn, dequantize, unpack_codes, logical_bytes, QuantizedLinearRTN
        self.q, self.dq, self.unpack, self.lb, self.QL = quantize_rtn, dequantize, unpack_codes, logical_bytes, QuantizedLinearRTN
        torch.manual_seed(0)

    def test_4_pack_unpack_round_trip_on_known_codes(self):
        codes = torch.tensor([[0, 15, 7, 8, 1, 14, 3, 12]], dtype=torch.uint8)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4))
        self.assertEqual(packed.tolist(), [[0xF0, 0x87, 0xE1, 0xC3]])
        torch.testing.assert_close(self.unpack(packed), codes)

    def test_4_quantize_error_bounded_by_half_scale(self):
        w = torch.randn(64, 256) * 0.05
        qd = self.q(w, 128)
        w_hat = self.dq(qd["packed"], qd["scale"], qd["zero"], 128, dtype=torch.float32)
        err = (w_hat - w).abs().reshape(64, 2, 128).amax(-1)
        half = qd["scale"].float() / 2
        self.assertTrue(bool((err <= half * 1.02 + 1e-6).all()))   # fp16 scale rounding slack
        self.assertEqual(qd["packed"].shape, (64, 128))
        self.assertEqual(qd["scale"].shape, (64, 2)); self.assertEqual(qd["zero"].shape, (64, 2))

    def test_4_exact_codes_round_trip(self):
        # a weight that is exactly representable (16 levels) must reconstruct exactly
        scale, zero = 0.25, 5
        codes = torch.randint(0, 16, (4, 128))
        w = (codes - zero).float() * scale
        qd = self.q(w, 128)
        w_hat = self.dq(qd["packed"], qd["scale"], qd["zero"], 128, dtype=torch.float32)
        torch.testing.assert_close(w_hat, w, atol=1e-6, rtol=0)

    def test_4_matmul_matches_dequantized_reference(self):
        w = torch.randn(32, 256).to(torch.bfloat16)
        qd = self.q(w, 128)
        m = self.QL(qd, out_dtype=torch.bfloat16)
        x = torch.randn(3, 256).to(torch.bfloat16)
        ref = torch.nn.functional.linear(x, m.dequantized())
        torch.testing.assert_close(m(x), ref)
        rel = float((m.dequantized().float() - w.float()).norm() / w.float().norm())
        self.assertLess(rel, 0.15)   # 4-bit RTN g128 on Gaussian weights lands near 0.10; this is a sanity bound only

    def test_11_logical_bytes_include_metadata(self):
        lb = self.lb(32768, 2048, 128)
        self.assertEqual(lb["data_bytes"], 32768 * 2048 // 2)
        self.assertEqual(lb["groups"], 32768 * 16)
        self.assertEqual(lb["metadata_bytes"], 32768 * 16 * 20 // 8)
        self.assertAlmostEqual(lb["effective_bits_per_element"], 4.15625)
        lb64 = self.lb(32768, 2048, 64)
        self.assertGreater(lb64["metadata_bytes"], lb["metadata_bytes"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestKVQuantizedLayer(unittest.TestCase):
    """Synthetic checks of transformers' KIVI-style QuantizedLayer (hqq backend, pure PyTorch), CPU."""

    def setUp(self):
        try:
            from transformers.cache_utils import HQQQuantizedLayer  # noqa: F401
            import hqq  # noqa: F401
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"transformers QuantizedLayer / hqq unavailable: {e}")

    def _layer(self, residual=4, axis=0):
        from transformers.cache_utils import HQQQuantizedLayer
        return HQQQuantizedLayer(nbits=4, axis_key=axis, axis_value=axis, q_group_size=64, residual_length=residual)

    def test_5_6_quantize_dequantize_known_values_and_append(self):
        torch.manual_seed(1)
        layer = self._layer(residual=4)
        k0 = torch.randn(1, 2, 8, 128); v0 = torch.randn(1, 2, 8, 128)
        k, v = layer.update(k0, v0)
        torch.testing.assert_close(k, k0); torch.testing.assert_close(v, v0)   # first call returns originals
        self.assertEqual(layer.get_seq_length(), 8)
        k1 = torch.randn(1, 2, 1, 128); v1 = torch.randn(1, 2, 1, 128)
        k, v = layer.update(k1, v1)
        self.assertEqual(k.shape, (1, 2, 9, 128)); self.assertEqual(layer.get_seq_length(), 9)
        # dequantized history approximates the original within 4-bit error
        err = (k[:, :, :8] - k0).abs().max() / k0.abs().max()
        self.assertLess(float(err), 0.2)
        torch.testing.assert_close(k[:, :, 8:], k1)   # residual (new token) returned exactly

    def test_7_gqa_shapes_preserved(self):
        layer = self._layer()
        k0 = torch.randn(1, 2, 16, 128); v0 = torch.randn(1, 2, 16, 128)
        layer.update(k0, v0)
        for _ in range(5):
            k, v = layer.update(torch.randn(1, 2, 1, 128), torch.randn(1, 2, 1, 128))
        self.assertEqual(k.shape[1], 2); self.assertEqual(k.shape[-1], 128); self.assertEqual(k.shape[2], 21)

    def test_requantization_drift_on_residual_flush(self):
        """Does re-quantizing dequantized history (every residual_length tokens) change old positions?
        axis 0 (transformers default) groups 64 elements strided across the flattened cache, so old positions are
        re-rounded against new group statistics; axis 1 groups 64 consecutive head_dim elements within a token."""
        drifts = {}
        for axis in (0, 1):
            torch.manual_seed(2)
            layer = self._layer(residual=4, axis=axis)
            k0 = torch.randn(1, 2, 8, 128); v0 = torch.randn(1, 2, 8, 128)
            layer.update(k0, v0)
            first = layer._dequantize(layer._quantized_keys).clone()
            for _ in range(8):   # forces two residual flushes
                layer.update(torch.randn(1, 2, 1, 128), torch.randn(1, 2, 1, 128))
            later = layer._dequantize(layer._quantized_keys)[:, :, :8]
            drifts[axis] = float((later - first).abs().max())
        print(f"\n[info] max drift of first 8 positions after two residual flushes: axis0={drifts[0]:.4g} axis1={drifts[1]:.4g} (0 = idempotent)")
        self.assertLess(drifts[1], 1e-6, "per-token grouping should re-quantize idempotently")
        self.assertGreaterEqual(drifts[0], 0.0)


@unittest.skipUnless(HAVE_TORCH and SLOW, "set FLEXTERN_SLOW_TESTS=1 with CUDA and the cached checkpoint")
class TestModelLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from common import load_model, verify_checkpoint
        cls.ident = verify_checkpoint()
        cls.model, cls.tok = load_model("cuda")

    def test_2_checkpoint_identity(self):
        self.assertTrue(self.ident["matches_adr_0004"])

    def test_1_model_loads_packed_and_generates(self):
        q = self.model.model.layers[0].self_attn.q_proj.weight
        self.assertEqual(q.dtype, torch.uint8); self.assertEqual(tuple(q.shape), (512, 2048))
        self.assertEqual(tuple(self.model.lm_head.weight.shape), (32768, 2048))
        ids = self.tok("The capital of France is", return_tensors="pt").input_ids.cuda()
        out = self.model.generate(ids, max_new_tokens=8, do_sample=False)
        self.assertGreater(out.shape[1], ids.shape[1])

    def test_8_incremental_dynamic_matches_single_forward(self):
        from evaluate import wikitext_windows, single_forward_nll, incremental_nll
        windows, _ = wikitext_windows(self.tok, 1, 256)
        ref = single_forward_nll(self.model, windows, 64, "cuda")
        inc = incremental_nll(self.model, windows, 64, "dynamic", "cuda")
        self.assertLess(abs(ref["mean_nll"] - inc["mean_nll"]), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
