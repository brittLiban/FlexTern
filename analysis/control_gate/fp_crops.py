"""Visual audit of the detections that land on vehicles (protocol_v2 / advisor item: label semantics vs detector error).

Samples detections >= tau whose best Person-GT IoU < 0.1 and whose best vehicle-GT IoU >= 0.3, evenly in time, crops each
(with margin) and draws: detection (red), vehicle GT (blue), Person GT (green). Writes one montage JPEG plus a JSON summary.
Usage: python -m analysis.control_gate.fp_crops CONFIG ROLE TAU [N]
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from analysis.edge_inference import evaluate as ev
from analysis.edge_inference.replay import _open
from . import run_cag


def main(argv=None):
    cfg, role, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
    n_pick = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    clip, video = run_cag.load_clip(role)
    dets, n = run_cag.load_dets(role, cfg, 3)
    g = clip.g
    cand = []
    keep = dets["score"] >= tau
    fr, bx, sc = dets["frame"][keep], dets["box"][keep], dets["score"][keep]
    order = np.argsort(fr, kind="stable")
    fr, bx, sc = fr[order], bx[order], sc[order]
    gmap = {}
    for i, f in enumerate(g["frame"]):
        gmap.setdefault(int(f), []).append(i)
    n_total = n_veh = n_person_overlap_inside_vehicle = 0
    for f, b, s in zip(fr, bx, sc):
        rows = gmap.get(int(f), [])
        gb = np.stack([[g["x1"][i], g["y1"][i], g["x2"][i], g["y2"][i]] for i in rows]) if rows else np.zeros((0, 4))
        cls = np.asarray([g["cls"][i] for i in rows])
        iou = ev.box_iou(b[None], gb)[0] if len(rows) else np.zeros(0)
        pi = iou[cls == 0].max() if (cls == 0).any() else 0.0
        vi = iou[(cls == 1) | (cls == 6)].max() if ((cls == 1) | (cls == 6)).any() else 0.0
        if pi < 0.1:
            n_total += 1
            if vi >= 0.3:
                n_veh += 1
                cand.append((int(f), b, float(s), pi, vi))
    if not cand:
        print("no vehicle-overlapping false positives"); return
    idx = np.linspace(0, len(cand) - 1, min(n_pick, len(cand))).astype(int)
    pick = [cand[i] for i in idx]
    want = {p[0] for p in pick}
    data = video.read_bytes()
    c, s = _open(data)
    frames = {}
    for i, fr_ in enumerate(c.decode(s)):
        if i in want:
            frames[i] = fr_.to_ndarray(format="rgb24")
        if i >= max(want):
            break
    c.close()
    thumbs, person_in_det = [], 0
    for f, b, sc_, pi, vi in pick:
        img = frames[f].copy()
        rows = gmap.get(f, [])
        for i in rows:
            col = (0, 255, 0) if g["cls"][i] == 0 else (0, 128, 255) if g["cls"][i] in (1, 6) else (200, 200, 200)
            cv2.rectangle(img, (int(g["x1"][i]), int(g["y1"][i])), (int(g["x2"][i]), int(g["y2"][i])), col, 3)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 4)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        half = int(max(b[2] - b[0], b[3] - b[1]) * 0.9) + 40
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        crop = img[y0:int(min(cy + half, 1080)), x0:int(min(cx + half, 1920))]
        crop = cv2.resize(crop, (256, 256))
        cv2.putText(crop, f"f{f} s{sc_:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        thumbs.append(crop)
    while len(thumbs) % 6:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows_ = [np.concatenate(thumbs[i:i + 6], 1) for i in range(0, len(thumbs), 6)]
    mont = np.concatenate(rows_, 0)
    out = run_cag.RESULTS / "fp_audit"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{role}_{cfg}_tau{tau:g}_vehicle_fp.jpg"
    cv2.imwrite(str(out / name), mont[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    summ = {"config": cfg, "role": role, "tau": tau, "false_positives_no_person_iou_ge_0.1": n_total,
            "of_which_overlap_vehicle_iou_ge_0.3": n_veh, "montage": name, "n_crops": len(pick),
            "legend": "red = detection, green = Person GT, orange = Forklift/PalletTruck GT"}
    (out / (name + ".json")).write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
