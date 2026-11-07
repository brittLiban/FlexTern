"""Amendment-1 diagnostic: can a single-frame encoder + task code localize the target? (supervised regression of the true
target position from the round-0 training frames; plain encoder vs task+coord-conditioned encoder). Result recorded in the
experiment record: plain median error 0.110 (22.6% under 0.05), conditioned 0.0175 (91.6% under 0.05). Needs the plain
(pre-amendment) architecture, which is reconstructed here so the diagnostic stays runnable."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import env as E
from . import models as M
from . import train as T


class PlainEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 2, 1); self.c2 = nn.Conv2d(16, 32, 3, 2, 1); self.c3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc = nn.Linear(1024, M.Z); self.ln = nn.LayerNorm(M.Z)

    def forward(self, x, task):
        x = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x))))))
        return self.ln(self.fc(x.flatten(1)))


def run(plain, path, steps=1500):
    d = np.load(path)
    sl = (slice(None), slice(None, None, 4))
    img, task, tgt = (torch.from_numpy(d[k][sl]).cuda() for k in ("img", "task", "target"))
    ok = torch.from_numpy(d["valid"][sl] & d["mask"][sl]).cuda().nonzero()
    tr, te = ok[: int(len(ok) * 0.9)], ok[int(len(ok) * 0.9):]
    torch.manual_seed(0)
    enc = (PlainEncoder() if plain else M.Encoder(3, "fp")).cuda()
    head = nn.Sequential(nn.Linear(M.Z + 6, 128), nn.ReLU(), nn.Linear(128, 2)).cuda()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), 2e-3)

    def fw(i):
        t = task[i[:, 0], i[:, 1]]
        z = enc(M.to_x(img[i[:, 0], i[:, 1]]), t)
        return head(torch.cat([z, t], -1))
    for _ in range(steps):
        b = tr[torch.randint(0, len(tr), (256,), device="cuda")]
        loss = F.mse_loss(fw(b), tgt[b[:, 0], b[:, 1]])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        b = te[:4000]
        e = (fw(b) - tgt[b[:, 0], b[:, 1]]).norm(dim=1)
    print("plain" if plain else "conditioned", "median err", float(e.median()), "frac<0.05", float((e < 0.05).float().mean()))


if __name__ == "__main__":
    p = T.gen_train(0)
    run(True, p)
    run(False, p)
