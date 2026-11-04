"""Tests for the FT-0010 control-adequacy gate (analysis/control_gate). numpy-only except where marked."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.control_gate import consistency, detectors, gt3d, zones


def _cam():
    # camera at origin looking along +z of the camera frame; world == camera frame for simplicity
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    E = np.hstack([np.eye(3), np.zeros((3, 1))])
    return {"K": K, "E": E}


class TestZones(unittest.TestCase):
    def test_zone_ids_row_major(self):
        m = zones.zone_masks(np.array([[10.0, 10.0], [1900.0, 1070.0], [960.0, 540.0]]))
        self.assertEqual(int(np.argmax(m[0])), 0)
        self.assertEqual(int(np.argmax(m[1])), 8)
        self.assertEqual(int(np.argmax(m[2])), 4)

    def test_margin_grows_zone(self):
        p = np.array([[645.0, 100.0]])   # 5 px beyond the x=640 boundary
        self.assertFalse(zones.zone_masks(p, 0.0)[0, 0])
        self.assertTrue(zones.zone_masks(p, 32.0)[0, 0])
        self.assertTrue(zones.zone_masks(p, 32.0)[0, 1])

    def test_gt_states_pos_neg_dontcare(self):
        foot = np.array([[100.0, 100.0], [650.0, 100.0]])      # zone 0 strictly; second straddles zone 0/1 boundary (inside zone 1)
        pos, neg = zones.gt_states(1, foot, np.array([0, 0]), np.array([100.0, 100.0]), np.array([True, True]))
        self.assertTrue(pos[0, 0] and pos[0, 1])
        pos2, neg2 = zones.gt_states(1, foot[1:], np.array([0]), np.array([100.0]), np.array([True]))
        self.assertFalse(pos2[0, 0])          # strictly in zone 1, not zone 0 ...
        self.assertFalse(neg2[0, 0])          # ... but within margin: don't care, not NEG
        self.assertTrue(neg2[0, 8])

    def test_small_and_invalid_persons_are_not_scored(self):
        foot = np.array([[100.0, 100.0], [100.0, 100.0]])
        pos, neg = zones.gt_states(1, foot, np.array([0, 0]), np.array([10.0, 100.0]), np.array([True, False]))
        self.assertFalse(pos.any())
        self.assertTrue(neg[0, 0])           # nothing scored there

    def test_det_states_ignore_short_and_low_score(self):
        d = {"frame": np.array([0, 0, 0]), "score": np.array([0.9, 0.2, 0.9], np.float32),
             "box": np.array([[90, 0, 110, 100], [90, 0, 110, 100], [1000, 500, 1010, 510]], np.float32)}
        pred = zones.det_states(1, d, 0.5)
        self.assertTrue(pred[0, 0])
        self.assertEqual(int(pred.sum()), 1)   # low-score and 10 px-tall detections ignored

    def test_constant_predictors_score_half(self):
        rng = np.random.default_rng(0)
        pos = rng.random((200, 9)) < 0.5
        neg = ~pos
        for const in (np.ones_like(pos), np.zeros_like(pos)):
            m = zones.occupancy_metrics(zones.occupancy_counts(pos, neg, const))
            self.assertAlmostEqual(m["balanced_accuracy"], 0.5)

    def test_perfect_predictor_and_bootstrap_seeded(self):
        rng = np.random.default_rng(1)
        pos = rng.random((900, 9)) < 0.4
        neg = ~pos
        m = zones.occupancy_metrics(zones.occupancy_counts(pos, neg, pos))
        self.assertEqual(m["balanced_accuracy"], 1.0)
        b1 = zones.block_bootstrap_occupancy(pos, neg, pos, n_boot=50)
        b2 = zones.block_bootstrap_occupancy(pos, neg, pos, n_boot=50)
        self.assertEqual(b1, b2)
        self.assertEqual(b1["balanced_accuracy_ci"], [1.0, 1.0])

    def test_presence_and_count(self):
        r = zones.presence_and_count(np.array([0, 1, 2, 3]), np.array([1, 1, 2, 5]))
        self.assertEqual(r["frames_without_scored_person"], 1)
        self.assertAlmostEqual(r["presence_false_alarm_rate"], 1.0)
        self.assertAlmostEqual(r["count_mae"], (1 + 0 + 0 + 2) / 4)


class TestGeometry(unittest.TestCase):
    def test_projection_pinhole(self):
        u, v, z = gt3d.project(np.array([[1.0, 0.5, 10.0]]), _cam())
        self.assertAlmostEqual(float(u[0]), 960 + 100)
        self.assertAlmostEqual(float(v[0]), 540 + 50)
        self.assertAlmostEqual(float(z[0]), 10.0)

    def test_box_corners_yaw(self):
        c = gt3d.box_corners(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([1.0]), np.array([2.0]),
                             np.array([np.pi / 2]))
        self.assertAlmostEqual(float(np.abs(c[0, :, 0]).max()), 0.5, places=6)   # x/y swapped by a 90 degree yaw
        self.assertAlmostEqual(float(np.abs(c[0, :, 1]).max()), 1.0, places=6)
        self.assertAlmostEqual(float(c[0, :, 2].min()), 0.0)
        self.assertAlmostEqual(float(c[0, :, 2].max()), 2.0)

    def test_visible_fraction(self):
        whole = np.array([[100.0, 100.0, 200.0, 300.0]])
        self.assertAlmostEqual(float(gt3d.visible_fraction(np.array([[100.0, 100.0, 200.0, 200.0]]), whole)[0]), 0.5)

    def test_consistency_rule(self):
        good = {"box_bottom": {"vis_h_over_expected_h_p5_25_50_75_95": [0.5, 0.9, 1.0, 1.05, 1.1],
                               "foot_v_minus_visible_bottom_px_p5_25_50_75_95": [-9, -5, -2, 0, 9],
                               "foot_u_minus_visible_centre_px_p5_25_50_75_95": [-5, -2, 1, 3, 9]}}
        self.assertTrue(consistency.passes(good))
        bad = json.loads(json.dumps(good))
        bad["box_bottom"]["vis_h_over_expected_h_p5_25_50_75_95"][2] = 0.78
        self.assertFalse(consistency.passes(bad))
        bad2 = json.loads(json.dumps(good))
        bad2["box_bottom"]["foot_v_minus_visible_bottom_px_p5_25_50_75_95"][2] = 40
        self.assertFalse(consistency.passes(bad2))
        self.assertFalse(consistency.passes({"box_bottom": {"n": 3}}))

    def test_person_geometry_recovers_standing_person(self):
        cam = _cam()
        # person 1.75 m tall, feet at world z=... use camera frame: feet at y=+1.0 (below axis), top at y=-0.75, depth 10 m
        # world->image: v = 540 + 1000*y/10. feet v=640, head v=465 => expected height 175 px
        g = {"cls": np.array([0]), "cx": np.array([0.0]), "cy": np.array([1.0]), "cz": np.array([10.0 + 0.0]),
             "sx": np.array([0.6]), "sy": np.array([0.6]), "sz": np.array([0.0]), "yaw": np.array([0.0]),
             "x1": np.array([940.0]), "y1": np.array([465.0]), "x2": np.array([980.0]), "y2": np.array([640.0]),
             "frame": np.array([0])}
        # in this toy frame "up" is -y, so project foot with z = depth: emulate by using rotated extrinsic
        E = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 1.0], [0, 1.0, 0, 0]])   # camera x=world x, camera y = -world z + 1, depth = world y
        cam = {"K": cam["K"], "E": E}
        g.update({"cy": np.array([10.0]), "cz": np.array([0.0])})
        m, vis, foot, eh, dz = consistency.person_geometry(g, cam, "z0")
        self.assertAlmostEqual(float(eh[0]), 175.0, delta=1.0)
        self.assertAlmostEqual(float(foot[0, 1]), 640.0, delta=1.0)
        self.assertGreater(float(dz[0]), 5)


class TestDetectorPostprocessing(unittest.TestCase):
    def test_yolox_grid_sizes(self):
        g, s = detectors.yolox_grid(416)
        self.assertEqual(len(g), 3549)       # (52^2 + 26^2 + 13^2), matches the ONNX output rows
        g6, _ = detectors.yolox_grid(640)
        self.assertEqual(len(g6), 8400)

    def test_yolox_post_decode_scale_and_nms(self):
        size = 416
        n = 3549
        out = np.zeros((1, n, 85), np.float32)
        out[:, :, 4] = 0.0
        # anchor index 0 is cell (0,0), stride 8: centre = (0+dx)*8, wh = exp(w)*8
        out[0, 0, :4] = [10.0, 10.0, np.log(5.0), np.log(10.0)]
        out[0, 0, 4], out[0, 0, 5] = 0.9, 0.8
        out[0, 1, :4] = [9.0, 10.0, np.log(5.0), np.log(10.0)]      # near duplicate -> suppressed by NMS
        out[0, 1, 4], out[0, 1, 5] = 0.8, 0.8
        post = detectors.yolox_post(size)
        boxes, scores = post(out, 1920, 1080)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(float(scores[0]), 0.72, places=5)
        r = min(size / 1080, size / 1920)
        cx, cy, w, h = 80.0, 80.0, 40.0, 80.0
        exp = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) / r
        np.testing.assert_allclose(boxes[0], exp, rtol=1e-4)

    def test_ssd_post_scales_to_frame(self):
        anchors = np.array([[100.0, 100.0, 200.0, 300.0], [0, 0, 10, 10]], np.float32)
        cls = np.full((2, 91), -10.0, np.float32)
        cls[0, 1] = 10.0
        cls[1, 0] = 10.0                  # anchor 1 is confidently background
        reg = np.zeros((2, 4), np.float32)
        post = detectors.ssd_post(anchors, 400, 400)
        boxes, scores = post((cls, reg), 1920, 1080)
        self.assertEqual(len(scores), 1)
        np.testing.assert_allclose(boxes[0], [100 * 4.8, 100 * 2.7, 200 * 4.8, 300 * 2.7], rtol=1e-4)


class TestHoldoutLock(unittest.TestCase):
    def test_holdout_locked_until_selection_exists(self):
        from analysis.control_gate import run_cag
        with tempfile.TemporaryDirectory() as d:
            old = run_cag.RESULTS
            run_cag.RESULTS = Path(d)
            try:
                with self.assertRaises(SystemExit):
                    run_cag.check_lock("hold_L08")
                run_cag.check_lock("dev_L10")             # development is never locked
                (Path(d) / "dev_selection.json").write_text("{}")
                run_cag.check_lock("hold_L08")
            finally:
                run_cag.RESULTS = old

    def test_clip_registry_layouts_disjoint_from_development(self):
        from analysis.control_gate import run_cag
        scenes = {r: v[1] for r, v in run_cag.CLIPS.items()}
        dev = {scenes[r] for r in scenes if r.startswith("dev_")}
        hold = {scenes[r] for r in scenes if r.startswith("hold_")}
        self.assertFalse(dev & hold)
        audit = json.loads((Path(__file__).resolve().parents[1] / "results" / "edge_inference" / "dataset_audit.json").read_text())
        grp = {s: g for g, ss in audit["layout_groups"].items() for s in ss}
        self.assertFalse({grp[s] for s in dev} & {grp[s] for s in hold})
        self.assertEqual(len({grp[s] for s in hold}), len(hold))


class TestMetricToleranceAsymmetry(unittest.TestCase):
    """Pins a property of the PREREGISTERED metric found in review: foot-jitter tolerance exists on the miss side only."""

    def test_preregistered_metric_counts_boundary_jitter_as_false_alarm(self):
        # GT foot 40 px left of the x=640 boundary (zone 0); detection foot 10 px closer to zone 1 (30 px left of boundary)
        foot = np.array([[600.0, 100.0]])
        pos, neg = zones.gt_states(1, foot, np.array([0]), np.array([100.0]), np.array([True]))
        d = {"frame": np.array([0]), "score": np.array([0.9], np.float32), "box": np.array([[580, 0, 620, 100]], np.float32)}
        d["box"][0] = [610, 0, 650, 100]       # foot x = 630: 10 px from the boundary, within 32 px of zone 1
        pred = zones.det_states(1, d, 0.5)
        self.assertTrue(neg[0, 1] and pred[0, 1])          # zone 1: NEG, predicted occupied -> counted as a false alarm
        self.assertEqual(zones.occupancy_counts(pos, neg, pred)["fp"], 1)

    def test_tolerant_variant_removes_it_and_counts_ignored_objects(self):
        foot = np.array([[600.0, 100.0]])
        pos, neg = zones.gt_states_tolerant(1, foot, np.array([0]), np.array([100.0]), np.array([True]))
        self.assertFalse(neg[0, 1])                          # 40 px from zone 1 < 2*margin: not a clean negative
        self.assertTrue(neg[0, 8])
        # an ignored (humanoid) foot deep in zone 4 makes zone 4 non-NEG
        pos, neg = zones.gt_states_tolerant(1, foot, np.array([0]), np.array([100.0]), np.array([True]),
                                            ignore_foot=np.array([[960.0, 540.0]]), ignore_frame=np.array([0]))
        self.assertFalse(neg[0, 4])
        self.assertFalse(pos[0, 4])


if __name__ == "__main__":
    unittest.main()
