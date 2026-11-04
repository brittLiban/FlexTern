"""
FT-0010 / EXP-0010 Stage A tests: scoring logic, shared pre/postprocessing, streaming-trace invariants,
timestamp preservation, ground-truth handling, layout-disjoint split, and run reproducibility.

Run from the repository root (host Python runs the pure-numpy subset; the av/cv2/torch/onnxruntime
tests skip cleanly when those are absent and run in the WSL venv):
    python -m unittest tests.test_edge_inference -v
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)

from analysis.edge_inference import common, dataset_audit, evaluate as ev, gt as gtmod, pipeline, replay  # noqa: E402


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAVE_AV, HAVE_CV2, HAVE_TORCH, HAVE_ORT, HAVE_IJSON = (_have("av"), _have("cv2"), _have("torch") and _have("torchvision"),
                                                       _have("onnxruntime"), _have("ijson"))
PROTO = common.load_protocol()


# --------------------------------------------------------------------------- scoring
class TestMatching(unittest.TestCase):
    def test_tp_fp_and_order(self):
        gt = np.array([[0, 0, 50, 100]], float)
        det = np.array([[1, 1, 51, 101], [200, 200, 260, 300]], float)
        flags, gms = ev.match_frame(det, np.array([0.9, 0.8]), gt, [False])
        self.assertEqual(flags.tolist(), [1, 0])
        self.assertAlmostEqual(gms[0], 0.9)

    def test_duplicate_detection_is_false_positive(self):
        gt = np.array([[0, 0, 50, 100]], float)
        det = np.array([[0, 0, 50, 100], [1, 0, 51, 100]], float)
        flags, _ = ev.match_frame(det, np.array([0.9, 0.8]), gt, [False])
        self.assertEqual(flags.tolist(), [1, 0])

    def test_ignore_gt_does_not_penalise_or_reward(self):
        gt = np.array([[0, 0, 10, 20]], float)  # small person -> ignore region
        det = np.array([[0, 0, 10, 20]], float)
        flags, gms = ev.match_frame(det, np.array([0.9]), gt, [True])
        self.assertEqual(flags.tolist(), [-1])
        self.assertEqual(gms[0], -1.0)

    def test_small_unmatched_detection_is_ignored_large_is_fp(self):
        gt = np.zeros((0, 4))
        det = np.array([[0, 0, 5, 20], [0, 0, 50, 120]], float)
        flags, _ = ev.match_frame(det, np.array([0.9, 0.8]), gt, [])
        self.assertEqual(flags.tolist(), [-1, 0])

    def test_prefers_non_ignored_gt(self):
        gt = np.array([[0, 0, 50, 100], [0, 0, 50, 100]], float)
        flags, gms = ev.match_frame(np.array([[0, 0, 50, 100]], float), np.array([0.7]), gt, [True, False])
        self.assertEqual(flags.tolist(), [1])
        self.assertEqual(gms.tolist(), [-1.0, 0.7])


class TestAPAndMetrics(unittest.TestCase):
    def test_ap_perfect_and_empty(self):
        self.assertAlmostEqual(ev.ap_from_arrays(np.array([0.9, 0.8]), np.array([1, 1], np.int8), None, 2), 1.0)
        self.assertEqual(ev.ap_from_arrays(np.zeros(0), np.zeros(0, np.int8), None, 3), 0.0)
        self.assertTrue(np.isnan(ev.ap_from_arrays(np.zeros(0), np.zeros(0, np.int8), None, 0)))

    def test_ap_known_value(self):
        # ranked: TP, FP, TP over 2 GT -> recall 0.5 @ p=1, recall 1.0 @ p=2/3
        ap = ev.ap_from_arrays(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1], np.int8), None, 2)
        expected = (51 * 1.0 + 50 * (2 / 3)) / 101  # recall thr 0..0.5 (51 pts) at 1.0, (0.5,1] (50 pts) at 2/3
        self.assertAlmostEqual(ap, expected, places=9)

    def test_ap_ignores_flag_minus_one(self):
        a = ev.ap_from_arrays(np.array([0.9, 0.85, 0.8]), np.array([1, -1, 1], np.int8), None, 2)
        self.assertAlmostEqual(a, 1.0)

    def _clip(self):
        n = 4
        gt = {"frame": np.array([0, 1, 2, 2], np.int32), "cls": np.array([0, 0, 0, 0], np.int32),
              "x1": np.array([0, 0, 0, 100], np.float32), "y1": np.zeros(4, np.float32),
              "x2": np.array([50, 50, 50, 110], np.float32), "y2": np.array([100, 100, 100, 20], np.float32)}  # last GT is 20 px tall -> ignore
        det = {"frame": np.array([0, 1, 3], np.int32), "box": np.array([[0, 0, 50, 100], [200, 0, 250, 100], [0, 0, 50, 100]], np.float32),
               "score": np.array([0.9, 0.8, 0.7], np.float32)}
        return ev.score_clip(n, det, gt)

    def test_operating_point_arithmetic(self):
        cs = self._clip()
        r = ev.operating_point(cs, 0.5)
        self.assertEqual((r["n_gt"], r["tp"], r["fp"], r["fn"]), (3, 1, 2, 2))  # frame 1 miss, frame 2 miss, FP on frames 1 and 3
        self.assertAlmostEqual(r["recall"], 1 / 3)
        self.assertAlmostEqual(r["precision"], 1 / 3)

    def test_dropped_frames_cost_recall(self):
        cs_all = self._clip()
        # same detections but frame 0 was dropped -> its TP disappears
        n = 4
        cs = cs_all
        keep = cs.det_frame != 0
        cs.det_frame, cs.det_score, cs.det_flag = cs.det_frame[keep], cs.det_score[keep], cs.det_flag[keep]
        cs.gt_matched_score = np.full_like(cs.gt_matched_score, -1.0)
        self.assertEqual(ev.operating_point(cs, 0.5)["recall"], 0.0)
        self.assertEqual(cs.n_frames, n)

    def test_best_f1_threshold(self):
        s = np.array([0.95, 0.6, 0.3, 0.2])
        f = np.array([1, 1, 0, 0], np.int8)
        cs = ev.ClipScore(4)
        cs.det_frame, cs.det_score, cs.det_flag = np.arange(4, dtype=np.int32), s, f
        cs.gt_frame, cs.gt_height, cs.gt_ignore = np.array([0, 1], np.int32), np.array([100., 100.]), np.zeros(2, bool)
        cs.gt_matched_score = np.array([0.95, 0.6])
        tau, rows = ev.best_f1_threshold(cs)
        self.assertEqual(rows[0]["tau"], 0.05)
        self.assertTrue(0.3 < tau <= 0.6)

    def test_bootstrap_is_seeded_and_block_based(self):
        rng = np.random.default_rng(0)
        n_fr = 1200
        cs = ev.ClipScore(n_fr)
        cs.det_frame = np.arange(n_fr, dtype=np.int32)
        cs.det_score = rng.random(n_fr)
        cs.det_flag = (rng.random(n_fr) < 0.6).astype(np.int8)
        cs.gt_frame = np.arange(n_fr, dtype=np.int32)
        cs.gt_height = np.full(n_fr, 100.0)
        cs.gt_ignore = np.zeros(n_fr, bool)
        cs.gt_matched_score = np.where(cs.det_flag == 1, cs.det_score, -1.0)
        a = ev.block_bootstrap(cs, 0.5, block=300, n_boot=200, seed=1)
        b = ev.block_bootstrap(cs, 0.5, block=300, n_boot=200, seed=1)
        c = ev.block_bootstrap(cs, 0.5, block=300, n_boot=200, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a["ap50_ci95"], c["ap50_ci95"])
        self.assertEqual(a["n_blocks"], 4)  # blocks of 300 frames, not 1200 independent samples

    def test_alignment_shift_sweep_detects_offset(self):
        n = 40
        gt_frames = np.arange(n, dtype=np.int32)
        gt = {"frame": gt_frames, "cls": np.zeros(n, np.int32), "x1": (gt_frames * 60).astype(np.float32),
              "y1": np.zeros(n, np.float32), "x2": (gt_frames * 60 + 40).astype(np.float32), "y2": np.full(n, 100, np.float32)}
        det_frame = gt_frames + 1  # detections lag GT by one frame
        det = {"frame": det_frame[:-1], "box": np.stack([gt["x1"][:-1], gt["y1"][:-1], gt["x2"][:-1], gt["y2"][:-1]], 1),
               "score": np.full(n - 1, 0.9, np.float32)}
        aps = {}
        for sh in (-1, 0, 1):
            cs = ev.score_clip(n - 3, det, gt, frame_shift=sh, frames=np.arange(2, n - 2))
            aps[sh] = ev.ap_from_arrays(cs.det_score, cs.det_flag, None, int((~cs.gt_ignore).sum()))
        self.assertEqual(max(aps, key=aps.get), -1)


# --------------------------------------------------------------------------- pipeline
class TestPipeline(unittest.TestCase):
    def test_nms_suppresses_overlap_keeps_best(self):
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        keep = pipeline.nms(b, np.array([0.5, 0.9, 0.4], np.float32), 0.55)
        self.assertEqual(keep.tolist(), [1, 2])

    def test_decode_zero_codes_returns_anchors(self):
        anc = np.array([[10, 20, 50, 100], [0, 0, 320, 320]], np.float32)
        np.testing.assert_allclose(pipeline.decode_boxes(np.zeros((2, 4), np.float32), anc), anc, atol=1e-4)

    def test_softmax_person_sums(self):
        logits = np.random.default_rng(0).normal(size=(5, 91)).astype(np.float32)
        p = pipeline.softmax_person(logits)
        self.assertTrue(((p > 0) & (p < 1)).all())

    def test_postprocess_scales_to_native_and_sorts(self):
        anc = np.array([[0, 0, 160, 160], [160, 160, 320, 320]], np.float32)
        logits = np.full((2, 91), -10.0, np.float32)
        logits[0, 1], logits[1, 1] = 5.0, 3.0
        boxes, scores = pipeline.postprocess(logits, np.zeros((2, 4), np.float32), anc, 1920, 1080)
        self.assertTrue((np.diff(scores) <= 0).all())
        np.testing.assert_allclose(boxes[0], [0, 0, 960, 540 / 1.0], atol=1e-2)

    def test_postprocess_empty(self):
        logits = np.zeros((3, 91), np.float32)
        logits[:, 1] = -30.0  # person probability ~ 1e-15, far below the 0.001 floor
        boxes, scores = pipeline.postprocess(logits, np.zeros((3, 4), np.float32),
                                             np.array([[0, 0, 10, 10]] * 3, np.float32), 1920, 1080)
        self.assertEqual((boxes.shape, scores.shape), ((0, 4), (0,)))

    @unittest.skipUnless(HAVE_CV2, "opencv not installed")
    def test_preprocess_shape_range_dtype_and_determinism(self):
        rgb = np.random.default_rng(0).integers(0, 256, (108, 192, 3), dtype=np.uint8)
        x = pipeline.preprocess(rgb)
        self.assertEqual((x.shape, x.dtype), ((1, 3, 320, 320), np.float32))
        self.assertTrue(x.min() >= -1.0 - 1e-6 and x.max() <= 1.0 + 1e-6)
        np.testing.assert_array_equal(x, pipeline.preprocess(rgb.copy()))
        white = pipeline.preprocess(np.full((108, 192, 3), 255, np.uint8))
        self.assertAlmostEqual(float(white.max()), 1.0, places=5)

    @unittest.skipUnless(HAVE_TORCH, "torch/torchvision not importable here")
    def test_decode_matches_torchvision_box_coder(self):
        import torch
        from torchvision.models.detection._utils import BoxCoder
        rng = np.random.default_rng(1)
        c = rng.uniform(40, 280, (200, 2)).astype(np.float32)
        wh = rng.uniform(10, 80, (200, 2)).astype(np.float32)
        anchors = np.concatenate([c - wh / 2, c + wh / 2], 1)
        rel = rng.normal(scale=2.0, size=(200, 4)).astype(np.float32)
        ref = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0)).decode_single(torch.from_numpy(rel), torch.from_numpy(anchors)).numpy()
        np.testing.assert_allclose(pipeline.decode_boxes(rel, anchors), ref, rtol=1e-4, atol=1e-3)

    @unittest.skipUnless(HAVE_TORCH, "torch/torchvision not importable here")
    def test_nms_matches_torchvision(self):
        import torch
        from torchvision.ops import nms as tv_nms
        rng = np.random.default_rng(2)
        c = rng.uniform(40, 280, (300, 2)).astype(np.float32)
        wh = rng.uniform(15, 90, (300, 2)).astype(np.float32)
        b = np.concatenate([c - wh / 2, c + wh / 2], 1)
        s = rng.random(300).astype(np.float32)
        ref = tv_nms(torch.from_numpy(b), torch.from_numpy(s), 0.55).numpy()
        np.testing.assert_array_equal(pipeline.nms(b, s, 0.55), ref)


# --------------------------------------------------------------------------- timestamps / traces
class TestTimestamps(unittest.TestCase):
    def test_frame_index_from_pts_tb(self):
        tb, fps = Fraction(1, 15360), Fraction(30)
        self.assertEqual([replay.frame_index(i * 512, tb, fps) for i in (0, 1, 2, 9269)], [0, 1, 2, 9269])
        self.assertEqual(replay.frame_index(1000 + 512 * 5, tb, fps, first_pts=1000), 5)


def _trace(n=10, drop=(), slow=()):
    t = replay._new_trace(n)
    period = 33_333_333
    for i in range(n):
        t["sched_ns"][i] = i * period
        t["release_ns"][i] = i * period + 10_000
        t["pts"][i] = i * 512
        if i in drop:
            t["dropped"][i] = 1
            t["done_ns"][i] = i * period + period
        else:
            t["dropped"][i] = 0
            t["start_ns"][i] = i * period + 1_000_000
            lat = 150_000_000 if i in slow else 20_000_000
            t["done_ns"][i] = i * period + lat
            t["infer_ms"][i], t["pre_ms"][i], t["post_ms"][i], t["decode_ms"][i] = 10.0, 1.0, 0.5, 3.0
    return t


class TestTraceLogic(unittest.TestCase):
    def test_valid_trace_passes(self):
        self.assertEqual(replay.validate_trace(_trace(drop=(3,))), [])

    def test_invalid_traces_are_caught(self):
        t = _trace()
        t["dropped"][4] = -1
        self.assertTrue(replay.validate_trace(t))
        t = _trace()
        t["done_ns"][2] = t["start_ns"][2] - 1
        self.assertTrue(any("done before start" in v for v in replay.validate_trace(t)))
        t = _trace()
        t["sched_ns"][5] = t["sched_ns"][4]
        self.assertTrue(any("strictly increasing" in v for v in replay.validate_trace(t)))

    def test_summary_counts_and_percentiles(self):
        t = _trace(n=100, drop=(7, 8), slow=(20, 21, 22))
        s = replay.summarize_trace(t, deadline_ms=100, wall_s=3.3)
        self.assertEqual((s["n_frames"], s["processed"], s["dropped"]), (100, 98, 2))
        self.assertEqual(s["deadline_missed_processed"], 3)
        self.assertAlmostEqual(s["on_time_fraction_of_all_frames"], 95 / 100)
        self.assertGreaterEqual(s["latency_ms"]["p99"], s["latency_ms"]["p95"])
        self.assertGreaterEqual(s["latency_ms"]["p95"], s["latency_ms"]["p50"])
        self.assertAlmostEqual(s["latency_ms"]["p50"], 20.0 - 0.0, places=0)


def _make_video(n=45, fps=30, w=96, h=64, seed=0):
    import av
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    c = av.open(buf, "w", format="mp4")
    s = c.add_stream("libx264", rate=fps, options={"bf": "0", "g": "10", "preset": "ultrafast"})
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :, 0] = (i * 5) % 255
        img += rng.integers(0, 8, img.shape, dtype=np.uint8)
        for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return buf.getvalue()


def _fake_engine(infer_s=0.0, calls=None):
    import time

    def pre(rgb):
        return rgb.astype(np.float32).mean(keepdims=True)

    def infer(x):
        if calls is not None:
            calls.append(float(x.ravel()[0]))
        if infer_s:
            time.sleep(infer_s)
        return x

    def post(out, w, h):
        return np.array([[1, 2, 3 + float(out.ravel()[0]) % 7, 4]], np.float32), np.array([0.9], np.float32)

    return replay.Engine("fake", pre, infer, post)


@unittest.skipUnless(HAVE_AV, "PyAV not installed (runs in the WSL venv)")
class TestStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = _make_video()

    def test_count_and_timestamps_preserved(self):
        n, tb, fps, first, w, h = replay.count_packets(self.video)
        self.assertEqual((n, w, h), (45, 96, 64))
        self.assertEqual(fps, Fraction(30))
        r = replay.run_offline(self.video, _fake_engine())
        tr = r["trace"]
        self.assertEqual(tr["frame"].tolist(), list(range(45)))
        step = np.diff(tr["pts"])
        self.assertTrue((step == step[0]).all() and step[0] > 0)  # constant, monotonic PTS preserved
        self.assertEqual(int(tr["dropped"].sum()), 0)

    def test_every_frame_logged_once_and_realtime_pacing(self):
        import time
        t = time.perf_counter()
        r = replay.run_replay(self.video, _fake_engine(0.002), do_warmup=2)
        wall = time.perf_counter() - t
        tr = r["trace"]
        self.assertEqual(replay.validate_trace(tr), [])
        self.assertEqual(int((tr["dropped"] == 0).sum()) + int((tr["dropped"] == 1).sum()), 45)
        self.assertEqual(int((tr["dropped"] == 1).sum()), 0)
        self.assertGreater(wall, 44 / 30 * 0.95)  # replay runs at the source frame rate, not faster
        self.assertLess(r["resources"]["wall_s"], 44 / 30 + 1.5)
        self.assertEqual(set(np.unique(r["dets"]["frame"]).tolist()), set(range(45)))

    def test_overload_drops_oldest_and_logs_them(self):
        r = replay.run_replay(self.video, _fake_engine(0.08), queue_cap=2, speed=4.0, do_warmup=1)
        tr = r["trace"]
        self.assertEqual(replay.validate_trace(tr), [])
        dropped = int((tr["dropped"] == 1).sum())
        self.assertGreater(dropped, 10)
        self.assertEqual(int((tr["dropped"] == 0).sum()) + dropped, 45)
        self.assertEqual(len(np.unique(r["dets"]["frame"])), int((tr["dropped"] == 0).sum()))  # drops produce no output
        s = replay.summarize_trace(tr, 100.0, r["resources"]["wall_s"])
        self.assertEqual(s["dropped"], dropped)

    def test_offline_is_deterministic(self):
        a = replay.run_offline(self.video, _fake_engine())["dets"]
        b = replay.run_offline(self.video, _fake_engine())["dets"]
        for k in ("frame", "box", "score"):
            np.testing.assert_array_equal(a[k], b[k])

    def test_replay_and_offline_see_identical_frames(self):
        calls_a, calls_b = [], []
        replay.run_offline(self.video, _fake_engine(0.0, calls_a))
        replay.run_replay(self.video, _fake_engine(0.0, calls_b), speed=2.0, do_warmup=0)
        self.assertEqual(calls_a, calls_b)  # same decode path -> same pixels reach the model


# --------------------------------------------------------------------------- ground truth
class TestGroundTruth(unittest.TestCase):
    def _write(self, key_style):
        k = (lambda a: a) if key_style == "space" else (lambda a: a.replace(" ", "_"))
        gt = {}
        for f in range(5):
            objs = []
            if f != 3:
                objs.append({k("object type"): "Person", k("object id"): 7, k("3d location"): [0, 0, 0],
                             k("2d bounding box visible"): {"Camera_0000": [10, 20, 50, 120], "Camera_0001": [1, 2, 3, 4]}})
            if f == 1:
                objs.append({k("object type"): "FourierGR1T2", k("object id"): 9, k("2d bounding box visible"): {"Camera_0000": [100, 100, 140, 200]}})
            gt[str(f)] = objs
        d = tempfile.mkdtemp()
        p = os.path.join(d, "ground_truth.json")
        json.dump(gt, open(p, "w"))
        return p

    @unittest.skipUnless(HAVE_IJSON, "ijson not installed")
    def test_both_key_styles_and_camera_filter(self):
        for style in ("space", "underscore"):
            cols, n, fp = gtmod.parse_ground_truth(self._write(style), {"Camera_0000"})
            self.assertEqual(set(cols), {"Camera_0000"})
            c = cols["Camera_0000"]
            self.assertEqual(c["frame"].tolist(), [0, 1, 1, 2, 4])  # frame 3 had no persons
            self.assertEqual(c["cls"].tolist(), [0, 0, 4, 0, 0])
            self.assertEqual(n, 5)
            self.assertEqual(gtmod.frame_coverage(fp, n)["frames_without_record"], 0)

    @unittest.skipUnless(HAVE_IJSON, "ijson not installed")
    def test_humanoid_boxes_are_ignore_regions_not_false_positives(self):
        cols, n, _ = gtmod.parse_ground_truth(self._write("space"), {"Camera_0000"})
        gt = cols["Camera_0000"]
        det = {"frame": np.array([1], np.int32), "box": np.array([[100, 100, 140, 200]], np.float32), "score": np.array([0.9], np.float32)}
        cs = ev.score_clip(n, det, gt)
        self.assertEqual(cs.det_flag.tolist(), [-1])  # firing on a humanoid is neither TP nor FP

    def test_content_hash_stable_and_sensitive(self):
        g = {"frame": np.arange(3, dtype=np.int32), "x1": np.arange(3, dtype=np.float32)}
        h1 = gtmod.content_sha256(g)
        self.assertEqual(h1, gtmod.content_sha256({k: v.copy() for k, v in g.items()}))
        g["x1"][1] += 1
        self.assertNotEqual(h1, gtmod.content_sha256(g))

    def test_size_stats(self):
        g = {"cls": np.zeros(4, np.int32), "frame": np.arange(4, dtype=np.int32), "x1": np.zeros(4, np.float32), "x2": np.full(4, 30, np.float32),
             "y1": np.zeros(4, np.float32), "y2": np.array([10, 40, 100, 200], np.float32)}
        s = gtmod.size_stats(g)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["frac_h_ge_32"], 0.75)


# --------------------------------------------------------------------------- dataset split integrity
class TestSplitIntegrity(unittest.TestCase):
    def test_layout_grouping_unions_geometry_or_map(self):
        sc = {n: {"geometry_sha256": g, "map_png_sha256": m} for n, g, m in
              [("A", "g1", "m1"), ("B", "g1", "m2"), ("C", "g3", "m2"), ("D", "g4", "m4")]}
        groups = dataset_audit.group_scenes(sc)
        self.assertEqual(sorted(map(tuple, groups.values())), [("A", "B", "C"), ("D",)])

    def test_protocol_roles_are_layout_disjoint(self):
        audit_path = os.path.join(_ROOT, "results", "edge_inference", "dataset_audit.json")
        if not os.path.exists(audit_path):
            self.skipTest("dataset audit not generated")
        audit = json.load(open(audit_path))
        groups = {}
        for role, c in PROTO["splits"].items():
            g = audit["scenes"][c["scene"]]["layout_group"]
            self.assertEqual(g, c["layout_group"], f"{role}: protocol layout group disagrees with audit")
            groups.setdefault(g, []).append(role)
        self.assertEqual(len(groups), 3, f"roles share a layout group: {groups}")

    def test_cosmos_rerenderings_share_layout_with_source(self):
        audit_path = os.path.join(_ROOT, "results", "edge_inference", "dataset_audit.json")
        if not os.path.exists(audit_path):
            self.skipTest("dataset audit not generated")
        audit = json.load(open(audit_path))
        self.assertEqual(audit["readme_ct25_pairs_not_confirmed_by_data"], [])
        for name, s in audit["scenes"].items():
            if s["ct25_of"]:
                self.assertEqual(s["layout_group"], audit["scenes"][s["ct25_of"]]["layout_group"])

    def test_protocol_is_frozen_and_complete(self):
        self.assertTrue(PROTO["frozen_before_any_model_run"])
        for k in ("iou_threshold", "size_policy", "operating_threshold", "dropped_frame_scoring", "statistics"):
            self.assertIn(k, PROTO["scoring"])
        self.assertIn("deadline_ms", PROTO["streaming"])
        self.assertIn("NOT MEASURED", PROTO["energy"])


# --------------------------------------------------------------------------- reproducibility
class TestReproducibility(unittest.TestCase):
    def test_environment_capture_does_not_import_torch(self):
        code = ("import sys; sys.path.insert(0, %r); from analysis.edge_inference import common; e = common.environment(); "
                "assert 'torch' not in sys.modules, 'environment() imported torch'; print(sorted(e))" % _ROOT)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for key in ("python", "numpy", "onnxruntime", "git", "platform"):
            self.assertIn(key, r.stdout)

    def test_manifest_and_identity_do_not_import_torch(self):
        """B1 (ORT) processes must stay torch-free; manifest() runs in them before timing starts."""
        script = chr(10).join([
            "import sys, os",
            "sys.path.insert(0, " + repr(_ROOT) + ")",
            "from analysis.edge_inference import models",
            "models.model_identity(); models.checkpoint_path()",
            "assert 'torch' not in sys.modules, 'model_identity() imported torch'",
            "from analysis.edge_inference import run_stage_a as R",
            "try:",
            "    R.manifest('development', 'B1', 'offline', 0)",
            "except FileNotFoundError:",
            "    pass  # clip files not present on this machine",
            "assert 'torch' not in sys.modules, 'manifest() imported torch'",
        ])
        code = script
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])

    def test_hashes_are_stable(self):
        self.assertEqual(common.sha256_json({"b": 1, "a": [1, 2]}), common.sha256_json({"a": [1, 2], "b": 1}))
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"abc")
        try:
            self.assertEqual(common.sha256_file(f.name), common.sha256_bytes(b"abc"))
        finally:
            os.unlink(f.name)

    @unittest.skipUnless(HAVE_ORT and HAVE_TORCH and HAVE_CV2, "needs torch + onnxruntime + opencv + exported ONNX")
    def test_torch_and_ort_agree_on_random_input(self):
        from analysis.edge_inference import models
        if not models.onnx_cache_path().exists() or not models.ANCHORS_PATH.exists():
            self.skipTest("run `run_stage_a prepare` first")
        m, _ = models.load_official(2)
        tr, ort_r = models.TorchRunner(m, 2), models.OrtRunner(models.onnx_cache_path(), 2)
        x = np.random.default_rng(0).uniform(-1, 1, (1, 3, 320, 320)).astype(np.float32)
        c0, b0 = tr(x)
        c1, b1 = ort_r(x)
        self.assertLess(float(np.abs(c0 - c1).max()), PROTO["parity_tolerances"]["raw_head_max_abs_diff_torch_vs_ort"])
        self.assertLess(float(np.abs(b0 - b1).max()), PROTO["parity_tolerances"]["raw_head_max_abs_diff_torch_vs_ort"])
        a1, a2 = ort_r(x), ort_r(x)
        np.testing.assert_array_equal(a1[0], a2[0])  # same input -> bit-identical output (determinism)


if __name__ == "__main__":
    unittest.main()
