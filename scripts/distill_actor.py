"""Distil the centralised planner into the decentralised actor (BC fine-tune).

Loads the (obs, actions) dataset produced by gen_planning_dataset.py and
fine-tunes each checkpoint's actor with supervised cross-entropy on the
planner's chosen actions.  The distilled actor is then decentralised: the
same local observation, one forward pass, no cloned simulator, no joint
search, no global state.  If even part of the planner's +0.021 survives,
the paper has a deployable claim instead of an oracle.

Run:
    python scripts/distill_actor.py --tag exp_ego_eval \
        --dataset ml_impl/runs/exp_ego_eval/planning_dataset.npz \
        --epochs 5 --lr 1e-4 --out-tag exp_distilled
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
from torch import nn

from uav_wireless.env import UAVIoTEnv
from uav_wireless.experiments import base_config
from uav_wireless.mappo import MAPPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_ego_eval")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out-tag", default="exp_distilled")
    ap.add_argument("--seeds", type=int, default=0, help="distil first K checkpoints (0 = all)")
    a = ap.parse_args()

    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.tag)
    ckpts = sorted(glob.glob(os.path.join(rundir, "seed_*.pt")))
    if a.seeds > 0:
        ckpts = ckpts[:a.seeds]
    if not ckpts:
        print("no checkpoints in", rundir)
        return 1
    dataset = a.dataset or os.path.join(rundir, "planning_dataset.npz")
    if not os.path.exists(dataset):
        print("dataset not found:", dataset)
        return 1

    d = np.load(dataset)
    obs_all = d["obs"].astype(np.float32)   # (T, N, obs_dim)
    acts_all = d["acts"].astype(np.int64)   # (T, N)
    # flatten to per-agent samples (o_i, a_i)
    obs_all = obs_all.reshape(-1, obs_all.shape[-1])
    acts_all = acts_all.reshape(-1)
    print("dataset: %d per-agent samples, obs_dim %d, n_actions %d"
          % (len(obs_all), obs_all.shape[1], acts_all.max() + 1))
    n = len(obs_all)

    outdir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.out_tag)
    os.makedirs(outdir, exist_ok=True)

    for ck in ckpts:
        meta = json.load(open(ck.replace(".pt", ".json"), encoding="utf-8"))
        cfg = base_config()
        cfg.ego_sensors = meta.get("ego", False)
        cfg.n_ego = meta.get("n_ego", 3)
        cfg.dist_shape_kappa = meta.get("shape", 0.0)
        cfg.use_w_penalty = meta.get("use_w", True)
        cfg.use_critical_anchor = meta.get("anchor", False)
        cfg.episode_len = meta.get("n_steps", 2000)
        env = UAVIoTEnv(cfg)
        agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                      state_dim=env.state_dim, device="cpu")
        agent.load(ck)

        actor = agent.actor
        opt = torch.optim.Adam(actor.parameters(), lr=a.lr)
        loss_fn = nn.CrossEntropyLoss()

        idx = np.arange(n)
        t0 = time.time()
        for epoch in range(a.epochs):
            np.random.shuffle(idx)
            total = 0.0
            nbatch = 0
            for start in range(0, n, a.batch):
                sel = idx[start:start + a.batch]
                obs_b = torch.as_tensor(obs_all[sel], dtype=torch.float32)
                act_b = torch.as_tensor(acts_all[sel], dtype=torch.int64)
                logits = actor(obs_b)  # (B, n_actions)
                loss = loss_fn(logits, act_b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                nbatch += 1
            print("  %s epoch %d: loss %.4f  [%.0fs]"
                  % (os.path.basename(ck), epoch + 1, total / nbatch,
                     time.time() - t0), flush=True)

        # save distilled checkpoint (same dict shape as MAPPO.save)
        name = os.path.basename(ck)
        out_pt = os.path.join(outdir, name)
        torch.save(dict(actor=actor.state_dict(), critic=agent.critic.state_dict(),
                        value_norm=agent.value_norm,
                        ret_mean=agent._ret_mean, ret_std=agent._ret_std),
                   out_pt)
        # copy the metadata json so eval_ckpt.py can restore the config
        import shutil
        shutil.copyfile(ck.replace(".pt", ".json"),
                        os.path.join(outdir, name.replace(".pt", ".json")))
        print("  distilled ->", out_pt, flush=True)

    print("\nDONE: distilled checkpoints in", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
