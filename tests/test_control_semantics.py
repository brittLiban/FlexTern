"""Tests for EXP-0010C (analysis/control_semantics): ground-plane geometry, ground-plane metric, provenance guard. numpy only."""
import unittest
from unittest import mock

import numpy as np

from analysis.control_semantics import gp_metric, ground, provenance


def toy_camera(height=5.0, tilt_deg=30.0, yaw_deg=0.0):
    """Camera at (0, 0, height) looking along +y tilted down; OpenCV convention (x right, y down, z forward)."""
    th, ya = np.radians(tilt_deg), np.radians(yaw_deg)
    fwd = np.array([np.sin(ya) * np.cos(th), np.cos(ya) * np.cos(th), -np.sin(th)])
    right = np.array([np.cos(ya), -np.sin(ya), 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    C = np.array([0.0, 0.0, height])
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    return {"K": K, "E": np.hstack([R, (-R @ C)[:, None]])}


class TestGround(unittest.TestCase):
    def test_roundtrip(self):
        cam = toy_camera()
        xy = np.array([[0.0, 8.0], [2.0, 12.0], [-3.0, 15.0]])
        uv, depth = ground.ground_to_pixel(xy, cam)
        back, d2, ok = ground.pixel_to_ground(uv, cam)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(back, xy, atol=1e-6)
        np.testing.assert_allclose(depth, d2, atol=1e-6)

    def test_ray_above_horizon_misses_plane(self):
        cam = toy_camera(tilt_deg=10.0)
        _, _, ok = ground.pixel_to_ground(np.array([[960.0, 0.0]]), cam)      # top of the image looks above the horizon
        self.assertFalse(ok[0])

    def test_one_row_covers_more_ground_far_away(self):
        cam = toy_camera()
        near = ground.metres_per_pixel_vertical(cam, 900)
        far = ground.metres_per_pixel_vertical(cam, 600)
        self.assertGreater(far, near)

    def test_visible_cells_contain_a_known_floor_point(self):
        cam = toy_camera()
        cells = ground.visible_cells(cam, 3.0, 0.5, 60.0)
        self.assertIn((0, 3), cells)         # (x=1.5, y=10.5) is mid-frame floor
        self.assertNotIn((0, -5), cells)     # behind the camera

    def test_cell_index(self):
        np.testing.assert_array_equal(ground.cell_index(np.array([[0.1, -0.1], [3.0, 5.9]]), 3.0), [[0, -1], [1, 1]])


def agents(cls, x, y, box=(900, 400, 940, 520), frame=0):
    n = len(x)
    z = np.zeros(n)
    return {"frame": np.full(n, frame, np.int32), "cls": np.asarray(cls, np.int32), "cx": np.asarray(x, float), "cy": np.asarray(y, float),
            "cz": np.full(n, 0.9), "sz": np.full(n, 1.8), "sx": np.full(n, 0.6), "sy": np.full(n, 0.6), "yaw": z,
            "x1": np.full(n, float(box[0])), "y1": np.full(n, float(box[1])), "x2": np.full(n, float(box[2])), "y2": np.full(n, float(box[3])),
            "oid": np.arange(n, dtype=np.int32)}


class TestGroundPlaneMetric(unittest.TestCase):
    def setUp(self):
        self.cam = toy_camera()
        self.cells = ground.visible_cells(self.cam, 3.0, 0.5, 60.0)
        self.col = {c: i for i, c in enumerate(self.cells)}

    def _pos_neg(self, g, positive):
        return gp_metric.gt_cell_states(1, self.cam, self.cells, g, positive)

    def test_person_pos_and_clean_neg(self):
        pos, neg, _ = self._pos_neg(agents([0], [1.5], [10.5]), (0,))
        c = self.col[(0, 3)]
        self.assertTrue(pos[0, c])
        far = self.col[(0, 6)] if (0, 6) in self.col else None
        if far is not None:
            self.assertTrue(neg[0, far])
        self.assertFalse(neg[0, c])

    def test_humanoid_is_negative_under_human_only_but_positive_under_agent_policy(self):
        g = agents([4], [1.5], [10.5])
        posA, negA, _ = self._pos_neg(g, gp_metric.POLICIES["A_human_only"])
        posB, negB, _ = self._pos_neg(g, gp_metric.POLICIES["B_person_shaped_agent"])
        c = self.col[(0, 3)]
        self.assertFalse(posA[0, c])
        self.assertTrue(negA[0, c])          # a detection on the humanoid is a false alarm for the human-only monitor
        self.assertTrue(posB[0, c])
        self.assertFalse(negB[0, c])

    def test_truncated_or_small_agent_is_ignored_not_positive_and_not_negative(self):
        g = agents([0], [1.5], [10.5], box=(900, 400, 940, 1080))       # touches the bottom edge
        pos, neg, info = self._pos_neg(g, (0,))
        c = self.col[(0, 3)]
        self.assertFalse(pos[0, c])
        self.assertFalse(neg[0, c])           # excluded from NEG too (ignored agent)
        self.assertEqual(info["agents_ignored"], 1)
        small = agents([0], [1.5], [10.5], box=(900, 400, 920, 420))     # 20 px tall
        pos, neg, _ = self._pos_neg(small, (0,))
        self.assertFalse(pos[0, c])
        self.assertFalse(neg[0, c])

    def test_detection_tolerance_matches_negative_exclusion(self):
        # an agent 0.6 m outside cell (0,3); a detection placed on it falls in the grown cell, so it is TP-compatible, not a false alarm
        g = agents([0], [3.6], [10.5])
        pos, neg, _ = self._pos_neg(g, (0,))
        uv, _ = ground.ground_to_pixel(np.array([[3.6, 10.5]]), self.cam)
        u, v = uv[0]
        dets = {"frame": np.array([0]), "score": np.array([0.9], np.float32),
                "box": np.array([[u - 20, v - 120, u + 20, v]], np.float32)}
        pred, info = gp_metric.det_cell_states(1, self.cam, self.cells, dets, 0.5)
        self.assertEqual(info["detections_used"], 1)
        c0, c1 = self.col[(0, 3)], self.col[(1, 3)]
        self.assertTrue(pred[0, c0] and pred[0, c1])          # within 0.75 m of both cells
        self.assertTrue(pos[0, c1])                            # agent is strictly in cell (1,3)
        self.assertFalse(neg[0, c0])                           # cell (0,3) is not a clean negative: agent within 1.5 m
        m = gp_metric.metrics_for(pos, neg, pred, np.array([0]))
        self.assertEqual(m["fp"], 0)

    def test_edge_touching_and_short_detections_are_excluded_and_counted(self):
        uv, _ = ground.ground_to_pixel(np.array([[1.5, 10.5]]), self.cam)
        u, v = uv[0]
        dets = {"frame": np.zeros(3, int), "score": np.full(3, 0.9, np.float32),
                "box": np.array([[u - 20, v - 120, u + 20, 1080.0],      # touches bottom edge
                                 [u - 20, v - 20, u + 20, v],             # 20 px tall
                                 [u - 20, v - 120, u + 20, v]], np.float32)}
        pred, info = gp_metric.det_cell_states(1, self.cam, self.cells, dets, 0.5)
        self.assertEqual(info["excluded_truncated_or_short"], 1)
        self.assertEqual(info["detections_used"], 1)

    def test_vehicle_detection_is_false_alarm(self):
        g = agents([1], [1.5], [10.5])                          # forklift only
        pos, neg, _ = self._pos_neg(g, (0, 4, 5))
        uv, _ = ground.ground_to_pixel(np.array([[1.5, 10.5]]), self.cam)
        u, v = uv[0]
        dets = {"frame": np.array([0]), "score": np.array([0.9], np.float32), "box": np.array([[u - 20, v - 120, u + 20, v]], np.float32)}
        pred, _ = gp_metric.det_cell_states(1, self.cam, self.cells, dets, 0.5)
        self.assertEqual(gp_metric.metrics_for(pos, neg, pred, np.array([0]))["fp"], 1)


class TestProvenance(unittest.TestCase):
    def test_guard_refuses_dirty_code(self):
        with mock.patch.object(provenance, "porcelain", return_value=[" M analysis/x.py"]):
            with self.assertRaises(SystemExit):
                provenance.require_clean_code()

    def test_guard_allows_clean_code_even_if_results_dirty(self):
        def fake(paths=None):
            return [] if paths else [" M results/x.json"]
        with mock.patch.object(provenance, "porcelain", side_effect=fake), mock.patch.object(provenance, "dirty_hash", return_value="h"):
            st = provenance.require_clean_code()
        self.assertTrue(st["dirty"])
        self.assertEqual(st["dirty_code_paths"], [])

    def test_repo_state_has_commit_and_scope(self):
        st = provenance.repo_state()
        self.assertEqual(len(st["commit"]), 40)
        self.assertIn("whole repository", st["scope"])
        self.assertIn("dirty_code_paths", st)


class FakeClip:
    def __init__(self):
        self.foot = np.array([[960.0, 800.0], [960.0, 800.0]])
        self.vis_h = np.array([200.0, 20.0])
        self.depth_ok = np.array([True, True])


class TestAttributionPriority(unittest.TestCase):
    """fa_attribution2.classify2: one fixed priority order (EXP-0010C, advisor review)."""

    def setUp(self):
        from analysis.control_semantics import fa_attribution2 as F
        self.F = F
        self.clip = FakeClip()

    def gt(self, boxes, classes):
        b = np.asarray(boxes, float)
        return {"x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3], "cls": np.asarray(classes)}

    def cls(self, det, gt, score=0.5, rows=None, row_of=None):
        rows = list(range(len(gt["cls"]))) if rows is None else rows
        return self.F.classify2(np.asarray(det, float), score, rows, gt, self.clip, row_of if row_of is not None else {0: 0})

    def test_humanoid_beats_person_and_vehicle(self):
        g = self.gt([[100, 100, 200, 300], [100, 100, 200, 300], [100, 100, 200, 300]], [4, 0, 1])
        self.assertEqual(self.cls([100, 100, 200, 300], g), "humanoid")

    def test_truncation_beats_person_and_clutter(self):
        g = self.gt([[0, 100, 100, 1080]], [0])
        self.assertEqual(self.cls([0, 500, 100, 1080], g), "truncated_detection")

    def test_person_scored_vs_ignored(self):
        g = self.gt([[100, 100, 200, 300]], [0])
        self.assertEqual(self.cls([100, 100, 200, 300], g), "person_scored")
        self.clip.vis_h = np.array([20.0, 20.0])
        self.assertEqual(self.cls([100, 100, 200, 300], g), "person_ignored")

    def test_partial_overlaps_are_split_not_lumped(self):
        # detection overlaps the object at IoU ~0.2: boxes 100x200 vs a box shifted so IoU is in [0.1, 0.3)
        det = [100, 100, 200, 300]
        shifted = [160, 100, 260, 300]            # IoU = 40*200/(2*20000-8000) = 0.25
        self.assertEqual(self.cls(det, self.gt([shifted], [4])), "humanoid_partial")
        self.assertEqual(self.cls(det, self.gt([shifted], [1])), "vehicle_or_amr_partial")
        self.assertEqual(self.cls(det, self.gt([shifted], [0])), "person_partial")
        self.assertEqual(self.cls(det, self.gt([det], [3])), "vehicle_or_amr")

    def test_clutter_vs_unexplained_by_score(self):
        g = self.gt([[900, 100, 950, 200]], [0])
        self.assertEqual(self.cls([100, 100, 200, 300], g, score=0.1), "low_score_clutter")
        self.assertEqual(self.cls([100, 100, 200, 300], g, score=0.6), "unexplained")


class TestFreshHoldoutLock(unittest.TestCase):
    def test_lock_required_committed_and_clean(self):
        import tempfile
        from pathlib import Path
        from analysis.control_semantics import run_cs
        with tempfile.TemporaryDirectory() as d:
            old = run_cs.LOCK
            run_cs.LOCK = Path(d) / "lock.json"
            try:
                with self.assertRaises(SystemExit):
                    run_cs.require_lock()                     # no lock file
                run_cs.LOCK.write_text('{"protocol_v3_sha256": "x"}')
                with self.assertRaises(SystemExit):
                    run_cs.require_lock()                     # protocol hash mismatch
            finally:
                run_cs.LOCK = old

    def test_fresh_layouts_are_disjoint_from_everything_used_before(self):
        import json
        from pathlib import Path
        sel = json.loads((Path(__file__).resolve().parents[1] / "results" / "control_semantics" / "fresh_selection.json").read_text())
        used_before = {"L00", "L02", "L08", "L09", "L10", "L11"}
        self.assertFalse({c["layout"] for c in sel["chosen"]} & used_before)
        self.assertEqual(len({c["layout"] for c in sel["chosen"]}), len(sel["chosen"]))


if __name__ == "__main__":
    unittest.main()
