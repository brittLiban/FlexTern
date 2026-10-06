"""
EXP-0005 validation tests: packing interpretation, Python reference, RTL engines, traffic model.

Run from the repository root:
    python -m unittest tests.test_hardware_gate -v

RTL tests compile and run the Icarus testbenches on vectors generated on the fly
(build/vectors/, seeded).  They are skipped when iverilog is not found
(FLEXTERN_IVERILOG_DIR, PATH, or C:\\iverilog\\bin).  Checkpoint-slice tests are
skipped when the pinned checkpoints are not in the local Hugging Face cache; they
never download.  Set FLEXTERN_FULL_RTL=1 to run every case at every engine
configuration (slower); by default a representative subset runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "hardware_gate"))
os.chdir(_ROOT)

from reference import (CANONICAL_K, accumulator_bits, decode_byte, decode_code, int4_accumulator_bits,  # noqa: E402
                       matvec_dense, matvec_from_packed, random_activations, random_ternary_packed, rowq_to_row,
                       ycore_code)
from safetensors_reader import pack_ternary, unpack_ternary  # noqa: E402
import vectors  # noqa: E402

RTL = [os.path.join("rtl", f) for f in ("y_core.sv", "ternary_unpack4.sv", "sum_tree.sv", "y_stream_engine.sv",
                                         "y_memory_bank.sv", "y_near_memory_engine.sv")]
VEC_DIR = os.path.join("build", "vectors")
FULL = os.environ.get("FLEXTERN_FULL_RTL") == "1"


def _iverilog_dir() -> str | None:
    cand = []
    if os.environ.get("FLEXTERN_IVERILOG_DIR"):
        cand.append(os.environ["FLEXTERN_IVERILOG_DIR"])
    w = shutil.which("iverilog")
    if w:
        cand.append(os.path.dirname(w))
    cand.append(r"C:\iverilog\bin")
    for d in cand:
        exe = os.path.join(d, "iverilog.exe" if os.name == "nt" else "iverilog")
        if os.path.isfile(exe):
            return d
    return None


IVDIR = _iverilog_dir()


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _run(cmd: list[str], timeout: int = 1200) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nexit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def _tool(name: str) -> str:
    return os.path.join(IVDIR, name + (".exe" if os.name == "nt" else ""))


# ----------------------------------------------------------------------------- packing / reference
class TestPackingInterpretation(unittest.TestCase):
    def test_1_2_3_code_values(self):
        self.assertEqual(decode_code(0b00), -1)
        self.assertEqual(decode_code(0b01), 0)
        self.assertEqual(decode_code(0b10), +1)

    def test_4_reserved_code_is_zero_and_documented(self):
        self.assertEqual(decode_code(0b11), 0)
        self.assertEqual(ycore_code(0b11), 0b10)             # Y-Core reserved -> 0 (ADR-0002)
        with open(os.path.join("rtl", "ternary_unpack4.sv"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("checkpoint 11 (unused) -> Y-Core 10 (reserved) -> contributes 0", src)

    def test_4_ycore_remap_is_decrement(self):
        for code, exp in ((0b00, 0b11), (0b01, 0b00), (0b10, 0b01), (0b11, 0b10)):
            self.assertEqual(ycore_code(code), exp)

    def test_5_four_weights_per_byte_matches_profiler_unpack(self):
        rng = np.random.default_rng(1)
        packed = random_ternary_packed(rng, 16, 64)
        q = unpack_ternary(packed)                         # [16, 64] via the EXP-0002 routine
        rows = packed.shape[0]
        for r in range(rows):
            for c in range(64):
                vals = decode_byte(int(packed[r, c]))
                for i in range(4):
                    self.assertEqual(vals[i], int(q[i * rows + r, c]))

    def test_6_output_dimension_order_vs_checkpoint_code(self):
        """Row produced by accumulator i of row-quad r is i*(N/4)+r; a byte at (r, c) is
        column c of rows {r, r+N/4, r+2N/4, r+3N/4}.  Mutation: swapping to r*4+i must fail."""
        rng = np.random.default_rng(2)
        packed = random_ternary_packed(rng, 32, 256)
        q = unpack_ternary(packed)
        x = random_activations(rng, 256)
        y = matvec_from_packed(packed, x)
        y_dense = matvec_dense(q, x)
        self.assertEqual(y, y_dense)
        n = 32
        mutated = [0] * n
        for r in range(n // 4):
            vals = [sum(int(decode_byte(int(packed[r, c]))[i]) * x[c] for c in range(256)) for i in range(4)]
            for i in range(4):
                mutated[r * 4 + i] = vals[i]                 # wrong order
                self.assertEqual(y[rowq_to_row(r, i, n)], vals[i])
        self.assertNotEqual(mutated, y)

    def test_roundtrip_pack_unpack(self):
        rng = np.random.default_rng(3)
        packed = random_ternary_packed(rng, 64, 2048)
        self.assertTrue(np.array_equal(pack_ternary(unpack_ternary(packed)), packed))

    def test_13_accumulator_widths_per_real_k(self):
        exp = {2048: 20, 2560: 20, 6912: 21, 9216: 22, 13312: 22}
        for k, bits in exp.items():
            self.assertEqual(accumulator_bits(k), bits, k)
            lim = 1 << (bits - 1)
            self.assertLess(k * 128, lim)                   # format extreme fits
            self.assertGreaterEqual(k * 128, lim >> 1)      # and needs this width (not one fewer)
        self.assertEqual(int4_accumulator_bits(13312), 25)

    def test_7_to_12_extremes_reference(self):
        k = 13312
        q = np.zeros((8, k), dtype=np.int8); q[0] = 1; q[1] = -1
        packed = pack_ternary(q)
        y = matvec_from_packed(packed, [-128] * k)
        self.assertEqual(y[0], -128 * k); self.assertEqual(y[1], 128 * k); self.assertEqual(y[2], 0)
        y = matvec_from_packed(packed, [127] * k)
        self.assertEqual(y[0], 127 * k); self.assertEqual(y[1], -127 * k)
        self.assertEqual(matvec_from_packed(packed, [0] * k), [0] * 8)


# ----------------------------------------------------------------------------- RTL
class _RtlBase(unittest.TestCase):
    manifest: dict = {}

    @classmethod
    def setUpClass(cls):
        if IVDIR is None:
            raise unittest.SkipTest("iverilog not found")
        os.makedirs(VEC_DIR, exist_ok=True)
        if not os.path.isfile(os.path.join(VEC_DIR, "manifest.json")):
            sys.argv = ["vectors.py", "--out", VEC_DIR, "--checkpoints"]
            vectors.main()
        with open(os.path.join(VEC_DIR, "manifest.json"), encoding="utf-8") as f:
            cls.manifest = json.load(f)["cases"]

    def compile_stream(self, bytes_: int, rowq: int, int4: bool = False) -> str:
        out = os.path.join("build", f"tb_{'int4' if int4 else 'stream'}_b{bytes_}_r{rowq}.vvp")
        srcs = [os.path.join("rtl", "sum_tree.sv"), os.path.join("rtl", "int4_stream_baseline.sv")] if int4 else RTL[:4]
        cmd = [_tool("iverilog"), "-g2012", f"-Py_stream_engine_tb.BYTES={bytes_}", f"-Py_stream_engine_tb.ROWQ={rowq}"]
        if int4:
            cmd.append("-DINT4_DUT")
        _run(cmd + ["-o", out] + srcs + [os.path.join("tests", "y_stream_engine_tb.sv")])
        return out

    def compile_nm(self, nb: int, bytes_: int) -> str:
        out = os.path.join("build", f"tb_nm_nb{nb}_b{bytes_}.vvp")
        _run([_tool("iverilog"), "-g2012", f"-Py_near_memory_engine_tb.NB={nb}", f"-Py_near_memory_engine_tb.BYTES={bytes_}",
              "-o", out] + RTL + [os.path.join("tests", "y_near_memory_engine_tb.sv")])
        return out

    def sim(self, vvp: str, case: str, mode: int, out: str | None = None, token: str = "y_stream_engine") -> str:
        args = [_tool("vvp"), vvp, f"+case={os.path.join(VEC_DIR, case)}", f"+mode={mode}"]
        if out:
            args.append(f"+out={out}")
        log = _run(args)
        self.assertIn(f"PASS: {token} verification completed.", log, log[-3000:])
        return log

    def case_or_skip(self, name: str):
        c = self.manifest.get(name)
        if c is None or c.get("skipped"):
            raise unittest.SkipTest(f"{name}: {c.get('skipped') if c else 'missing'}")
        return c


class TestStreamingEngineRTL(_RtlBase):
    def test_14_to_18_all_real_k_random_and_extremes(self):
        vvp = self.compile_stream(64, 1)
        ks = CANONICAL_K if FULL else (2048, 2560, 13312)
        for k in ks:
            log = self.sim(vvp, f"rand_k{k}", 0)
            self.assertIn("II=1 check ok=1", log)
            self.sim(vvp, f"ext_k{k}", 0)                   # items 7-13: extremes, no overflow

    def test_22_ii1_and_21_backpressure(self):
        vvp = self.compile_stream(64, 1)
        log0 = self.sim(vvp, "rand_k2048", 0, out=os.path.join("build", "ii_m0.txt"))
        self.assertIn("II=1 check ok=1", log0)
        log1 = self.sim(vvp, "rand_k2048", 1, out=os.path.join("build", "ii_m1.txt"))
        self.assertIn("stall_violations=0", log1)
        self.assertEqual(_read(os.path.join("build", "ii_m0.txt")), _read(os.path.join("build", "ii_m1.txt")))

    def test_4_reserved_code_contributes_zero(self):
        vvp = self.compile_stream(64, 1)
        self.sim(vvp, "reserved_k2048", 0)

    def test_19_20_multi_row_streams_and_tail(self):
        vvp = self.compile_stream(16, 4)                   # 4 row streams x 16 bytes = 256 terms
        self.sim(vvp, "tail_k2048", 0)                      # 13 row-quads: last group has 1 valid stream
        self.sim(vvp, "tail_k2048", 1)
        vvp = self.compile_stream(32, 8)
        self.sim(vvp, "rand_k2048", 1)

    def test_23_24_consecutive_ops_and_reset(self):
        # every case runs n_ops >= 2 back to back; extremes run 4 ops with the accumulators
        # saturating in both directions, which proves the per-op reset of the accumulators
        vvp = self.compile_stream(64, 1)
        log = self.sim(vvp, "ext_k2048", 0)
        self.assertEqual(log.count("II=1 check ok=1"), 4)

    def test_26_checkpoint_slices(self):
        vvp = self.compile_stream(64, 1)
        names = ["ckpt_f1b_k_proj", "ckpt_f1b_down", "ckpt_f3b_down", "ckpt_bn_q_proj", "ckpt_bn_down"]
        ran = 0
        for n in names:
            try:
                self.case_or_skip(n)
            except unittest.SkipTest:
                continue
            self.sim(vvp, n, 0)
            ran += 1
        if ran == 0:
            raise unittest.SkipTest("no checkpoint slices available locally")


class TestNearMemoryEngineRTL(_RtlBase):
    def test_25_identical_to_streaming(self):
        vs = self.compile_stream(64, 1)
        vn = self.compile_nm(4, 16)
        cases = ["rand_k2048", "rand_k13312", "tail_k2048"] + (["ckpt_f1b_k_proj"] if "ckpt_f1b_k_proj" in self.manifest and not self.manifest["ckpt_f1b_k_proj"].get("skipped") else [])
        for c in cases:
            a = os.path.join("build", f"{c}_stream.txt"); b = os.path.join("build", f"{c}_nm.txt")
            self.sim(vs, c, 0, out=a)
            self.sim(vn, c, 0, out=b, token="y_near_memory_engine")
            self.assertEqual(_read(a), _read(b), c)

    def test_near_memory_backpressure_and_tail_banks(self):
        vn = self.compile_nm(8, 16)
        self.sim(vn, "tail_k2048", 1, token="y_near_memory_engine")   # 13 rows over 8 banks: 2 per bank, last banks idle
        self.sim(vn, "ext_k2048", 0, token="y_near_memory_engine")     # 2 rows over 8 banks: 6 banks idle
        self.sim(vn, "reserved_k2048", 0, token="y_near_memory_engine")

    def test_near_memory_all_k(self):
        vn = self.compile_nm(4, 32)
        for k in (CANONICAL_K if FULL else (2560, 6912, 9216)):
            self.sim(vn, f"rand_k{k}", 0, token="y_near_memory_engine")


class TestInt4BaselineRTL(_RtlBase):
    def test_int4_baseline_correct(self):
        vvp = self.compile_stream(128, 1, int4=True)
        for c in ("int4_k2048", "int4_k13312"):
            log = self.sim(vvp, c, 0)
            self.assertIn("II=1 check ok=1", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ----------------------------------------------------------------------------- analytical models
class TestWorkloadAndTrafficModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import workload_mapping as wm
        import traffic_model as tm
        cls.wm, cls.tm = wm, tm
        cls.ws = wm.load_workloads()

    def test_four_terms_per_byte_invariant(self):
        for slug, w in self.ws.items():
            for t in self.wm.tensors(w):
                self.assertEqual(t["terms"], 4 * t["packed_bytes"], (slug, t["module"]))
            self.assertEqual(sum(t["packed_bytes"] * t["layers"] for t in self.wm.tensors(w)), w.W // 4)

    def test_real_k_set(self):
        ks = sorted({t["K"] for w in self.ws.values() for t in self.wm.tensors(w)})
        self.assertEqual(ks, list(CANONICAL_K))

    def test_stream_cycle_law_matches_simulated_case(self):
        # simulated: K=2048, N=64 (16 row-quads), B=64 -> 512 chunks (II=1, +2 latency measured in RTL)
        t = {"K": 2048, "N": 64, "rowq": 16, "terms": 64 * 2048}
        c, u = self.wm.stream_cycles(t, 64, 1)
        self.assertEqual(c, 512); self.assertEqual(u, 1.0)
        c, u = self.wm.nearmem_cycles({"K": 2048, "rowq": 16, "terms": 64 * 2048}, 4, 16)[:2]
        self.assertEqual(c, 512)                            # rpb 4 x 128 chunks (RTL measured 519 incl. latency)

    def test_tail_utilization(self):
        t = {"K": 2048, "N": 256, "rowq": 64, "terms": 256 * 2048}     # k_proj
        c, u = self.wm.stream_cycles(t, 1, 256)
        self.assertEqual(u, 0.25)                                       # 64 row streams of 256 busy
        c, u, cap = self.wm.nearmem_cycles(t, 256, 16)
        self.assertEqual(c, 128); self.assertEqual(u, 0.25); self.assertEqual(cap, 2048)
        # collector bound: 64 banks x 64 B on K=2048 would finish in 32 cycles but must emit 64 groups
        c, u, cap = self.wm.nearmem_cycles(t, 64, 64)
        self.assertEqual(c, 64); self.assertEqual(u, 0.5)

    def test_required_clock_is_bandwidth_over_bytes_per_cycle(self):
        for r in self.wm.bandwidth_rows():
            self.assertAlmostEqual(r["required_clock_MHz_for_100GBps"], 100e9 / r["packed_bytes_per_cycle"] / 1e6, places=1)
            self.assertIn("UNMEASURED", r["status"])

    def test_traffic_levels_consistent(self):
        for slug, w in self.ws.items():
            r = self.tm.per_token(w, 64)
            self.assertEqual(r["nm_C_local_reads"], r["stream_B_global_weights"])      # local reads do not vanish
            self.assertEqual(r["nm_B_global_weights"], 0)
            self.assertEqual(r["nm_A_per_token_if_per_layer_residency"], r["stream_B_global_weights"])
            self.assertEqual(r["nm_D_activation_delivered"], 64 * r["nm_D_activation_injected"])
            lock = self.tm.per_token(w, 64, buffered=False)
            self.assertEqual(lock["nm_D_activation_delivered"], lock["packed_weight_bytes"])   # lockstep = weight bytes
            i4 = self.tm.per_token(w, 64, weight_bits=4)
            self.assertEqual(i4["nm_C_local_reads"], 2 * r["nm_C_local_reads"])
            self.assertEqual(i4["nm_B_global_per_token"], r["nm_B_global_per_token"])      # near-memory gain is format-neutral

    def test_reconnect_reproduces_exp0003_reference_total(self):
        rows = [r for r in self.tm.reconnect_rows(self.ws) if r["model"] == "falcon-e-1b" and r["context"] == 2048
                and r["scenario"].startswith("BF16")]
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["global_total_streaming"] / 1e6, 581.1, places=0)     # EXP-0003 summary row
        self.assertEqual(rows[0]["limiter_near_memory"], "head")

    def test_capacity_rows(self):
        rows = [r for r in self.tm.capacity_rows(self.ws) if r["model"] == "falcon-e-1b"]
        self.assertEqual(rows[0]["packed_bytes_total"], 396361728)
        r1m = [r for r in rows if r["bank_capacity_bytes"] == 1 << 20][0]
        self.assertEqual(r1m["banks_needed_whole_model"], 378)
