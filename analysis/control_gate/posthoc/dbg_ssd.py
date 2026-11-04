import numpy as np, torch
from pathlib import Path
from analysis.edge_inference import models as M
from analysis.control_gate import detectors as D, parity as P
from analysis.edge_inference.common import cache_dir
name = "S1"
w, h = D.SSD_SIZES[name]
model, _ = M.load_official(threads=4)
model.transform.fixed_size = (w, h)
print("transform", model.transform)
video = cache_dir() / "raw/train/Warehouse_017/videos/Camera_0001.mp4"
fr = P._frames(video, [225, 450])
rgb = fr[225]
t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
with torch.inference_mode():
    il, _ = model.transform([t])
    print("tensor shape", il.tensors.shape, il.image_sizes)
    feats = model.backbone(il.tensors)
    feats = list(feats.values())
    ho = model.head(feats)
    an = model.anchor_generator(il, feats)[0]
print("anchors official", an.shape, "mine", np.load(D.ssd_paths(name)[1]).shape)
mine_an = np.load(D.ssd_paths(name)[1])
print("anchor max diff", np.abs(an.numpy() - mine_an).max() if an.shape == mine_an.shape else "shape mismatch")
eng, runner = D.build_engine(name, 4)
x = eng.pre(rgb)
print("input diff (mine vs torch-resized)", np.abs(x - il.tensors.numpy()).max(), np.abs(x - il.tensors.numpy()).mean())
c, b = eng.infer(x)
print("cls diff with torch resized input:", np.abs(c - ho["cls_logits"][0].numpy()).max())
c2, b2 = eng.infer(il.tensors.numpy())
print("ORT on torch-resized input vs torch head: ", np.abs(c2 - ho["cls_logits"][0].numpy()).max(), np.abs(b2 - ho["bbox_regression"][0].numpy()).max())
boxes, scores = eng.post((c2, b2), rgb.shape[1], rgb.shape[0])
with torch.inference_mode():
    off = model([t])[0]
m = (off["labels"] == 1) & (off["scores"] >= 0.3)
print("official n", int(m.sum()), "mine(torch-resized input) n", int((scores >= 0.3).sum()), "mine(cv2 input)", int((eng.post((c, b), rgb.shape[1], rgb.shape[0])[1] >= 0.3).sum()))
print(off["boxes"][m][:3], boxes[scores >= 0.3][:3])
