"""Visual + numeric alignment check between decoded video frames and ground-truth boxes.

Draws the GT person boxes on selected decoded frames so an off-by-one / wrong-camera
error is visible *before* any model is scored. Also verifies PTS monotonicity and that
the decoded frame count equals the GT frame count.

Usage: python -m analysis.edge_inference.alignment_check SCENE CAMERA SPLIT FRAME [FRAME...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .common import RESULTS, cache_dir
from .gt import load_camera_gt


def decode_frames(video: Path, wanted: set[int]):
    import av
    out, n = {}, 0
    pts = []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            n += 1
            pts.append(fr.pts)
            if i in wanted:
                out[i] = fr.to_ndarray(format="rgb24")
    return out, n, np.asarray(pts, dtype=np.int64)


def main(argv):
    import cv2
    scene, cam, split = argv[0], argv[1], argv[2]
    frames = [int(x) for x in argv[3:]]
    video = cache_dir() / "raw" / split / scene / "videos" / f"{cam}.mp4"
    gt = load_camera_gt(cache_dir() / "gt" / scene / f"{cam}.npz")
    imgs, n, pts = decode_frames(video, set(frames))
    d = np.diff(pts)
    print(f"{scene}/{cam}: decoded {n} frames, GT n_frames {gt['n_frames']}, pts steps {np.unique(d).tolist()}, monotonic {bool((d > 0).all())}")
    outdir = RESULTS / "alignment"
    outdir.mkdir(parents=True, exist_ok=True)
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 255, 0), 3: (255, 0, 255), 6: (0, 255, 255)}
    for f in frames:
        img = cv2.cvtColor(imgs[f], cv2.COLOR_RGB2BGR)
        m = gt["frame"] == f
        for x1, y1, x2, y2, c in zip(gt["x1"][m], gt["y1"][m], gt["x2"][m], gt["y2"][m], gt["cls"][m]):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors.get(int(c), (255, 255, 255)), 3)
        cv2.putText(img, f"{scene}/{cam} frame {f}  GT boxes {int(m.sum())} (green=Person)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        p = outdir / f"{scene}_{cam}_f{f:05d}.jpg"
        cv2.imwrite(str(p), cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1:])
