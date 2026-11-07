"""Amendment 1 (EXP-0012): DAgger-style data aggregation. A round-0 FP16 recurrent policy (G, seed 0) is rolled out with random
expert intervention; the privileged expert relabels every visited state. The aggregated set is then shared by ALL arms.
Reason: round-0 BC (clean+noisy expert demonstrations only) gave ~10-20% closed-loop success for every arm, a floor effect
caused by compounding errors, not by memory (documented in the experiment record)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from . import env as E
from . import expert as X
from . import models as M
from . import train as T


@torch.no_grad()
def collect(net, n_episodes, seed, Tn=T.T_TRAIN, betas=(0.0, 0.3, 0.6), batch=250, dev="cuda"):
    rng = np.random.default_rng(seed)
    conds_all = T.train_conds()
    out = {k: [] for k in ("img", "valid", "prop", "task", "a_exec", "a_label", "mask")}
    for b0 in range(0, n_episodes, batch):
        nb = min(batch, n_episodes - b0)
        cs = [conds_all[int(rng.integers(len(conds_all)))] for _ in range(nb)]
        env = E.BatchEnv(nb, "train", cs, seed=int(rng.integers(1 << 31)))
        beta = np.array([betas[int(rng.integers(len(betas)))] for _ in range(nb)], np.float32)
        obs = env.reset()
        h = net.init_state(nb, dev)
        a_prev = torch.zeros(nb, 3, device=dev)
        buf = {k: [] for k in out}
        for t in range(Tn):
            alive = ~env.done
            img = torch.from_numpy(obs["img"]).to(dev)
            prop = torch.from_numpy(obs["prop"]).to(dev)
            task = torch.from_numpy(obs["task"]).to(dev)
            valid = torch.from_numpy(obs["valid"]).float().to(dev)
            z = net.enc(M.to_x(img), task)
            h = net.cell_step(h, z, valid, prop, task, a_prev)
            lg = net.policy(h, prop, task, z * valid[:, None])
            pa = torch.cat([torch.tanh(lg[:, :2]), (lg[:, 2:3] > 0).float()], -1).cpu().numpy()
            lab = X.expert_action(env)
            use_exp = rng.random(nb) < beta
            ex = np.where(use_exp[:, None], lab, pa).astype(np.float32)
            ex[:, :2] = np.clip(ex[:, :2] + rng.normal(0, 0.1, (nb, 2)).astype(np.float32), -1, 1)
            for k, v in (("img", obs["img"]), ("valid", obs["valid"]), ("prop", obs["prop"]), ("task", obs["task"]),
                         ("a_exec", ex), ("a_label", lab), ("mask", alive)):
                buf[k].append(v)
            a_prev = torch.from_numpy(ex).to(dev)
            obs, _ = env.step(ex)
        for k in out:
            out[k].append(np.stack(buf[k], 1))
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def main():
    ck = sys.argv[1]
    n = int(sys.argv[2])
    seed = int(sys.argv[3])
    out = os.path.join(T.DATA_DIR, f"dagger_s{seed}_n{n}.npz")
    net, _ = T.load_net(ck)
    d = collect(net, n, seed)
    np.savez(out, **d)
    print("wrote", out, {k: v.shape for k, v in d.items()})


if __name__ == "__main__":
    main()
