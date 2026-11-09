"""Generate a behaviour-cloning dataset from the centralised planner.

For each (checkpoint, scene) the planner of eval_planning_v2.py runs: at
each decision step, sample K candidate joint actions from the actor, roll
each H steps in a cloned env, pick the best by cumulative CK.  For the
WINNING candidate we record the per-agent experiences (local obs, chosen
action).  The result is a dataset of (o_i, a_i) pairs that a decentralised
actor can be distilled on: the planner's choices become the target policy,
and the distilled actor inherits the lookahead gain without any global state
at deployment.

Only a modest number of scenes/checkpoints is needed (the actor is already
close to the target), so this runs in minutes, not hours.

Run:
    python scripts/gen_planning_dataset.py --tag exp_ego_eval \
        --eval-offset 1000 --eval-seeds 10 --steps 500 --H 2 --K 4 \
        --out ml_impl/runs/exp_ego_eval/planning_dataset.npz
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

from uav_wireless.env import UAVIoTEnv
from uav_wireless.experiments import base_config
from uav_wireless.mappo import MAPPO


def action_masks_safe(env):
    mask = np.stack(env.action_masks())
    for i in range(env.cfg.N):
        if mask[i].sum() == 0:
            mask[i, 0] = True
    return mask


def rollout_score(env, agent, cand, H):
    env2 = copy.deepcopy(env)
    sc = 0.0
    a = cand
    for h in range(H):
        obs2, _, r, done, _ = env2.step(a)
        sc += float(env2.summary().get("CK", 0.0))
        if done:
            break
        mask = action_masks_safe(env2)
        with torch.no_grad():
            a = agent.act(torch.as_tensor(obs2, dtype=torch.float32),
                           torch.as_tensor(mask, dtype=torch.bool),
                           deterministic=True)
    return sc


def collect(agent, cfg, steps, seed, H, K):
    """Return (obs, acts) arrays for one episode under the planner."""
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = UAVIoTEnv(c)
    obs, _ = env.reset()
    obs_all, act_all = [], []
    n = 0
    while n < steps:
        mask = action_masks_safe(env)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits = agent.masked_logits(obs_t, mask_t)
            logp = torch.log_softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(logits=logp)
        best_acts = None
        best_score = -1e18
        for _ in range(K):
            cand = dist.sample().cpu().numpy()
            sc = rollout_score(env, agent, cand, H)
            if sc > best_score:
                best_score = sc
                best_acts = cand
        if best_acts is None:
            with torch.no_grad():
                best_acts = agent.act(obs_t, mask_t, deterministic=True)
        obs_all.append(obs)
        act_all.append(best_acts)
        obs, _, r, done, _ = env.step(best_acts)
        n += 1
        if done:
            break
    return np.asarray(obs_all), np.asarray(act_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_ego_eval")
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.tag)
    ckpts = sorted(glob.glob(os.path.join(rundir, "seed_*.pt")))
    if not ckpts:
        print("no checkpoints in", rundir)
        return 1
    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    print("planning dataset H=%d K=%d on %d scenes x %d ckpts, %d steps"
          % (a.H, a.K, len(scenes), len(ckpts), a.steps), flush=True)

    obs_all, act_all = [], []
    t0 = time.time()
    for ck in ckpts:
        meta = json.load(open(ck.replace(".pt", ".json"), encoding="utf-8"))
        cfg = base_config()
        cfg.ego_sensors = meta.get("ego", False)
        cfg.n_ego = meta.get("n_ego", 3)
        cfg.dist_shape_kappa = meta.get("shape", 0.0)
        cfg.use_w_penalty = meta.get("use_w", True)
        cfg.use_critical_anchor = meta.get("anchor", False)
        cfg.episode_len = a.steps
        env = UAVIoTEnv(cfg)
        agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                      state_dim=env.state_dim, device="cpu")
        agent.load(ck)
        for s in scenes:
            o, ac = collect(agent, cfg, a.steps, s, a.H, a.K)
            obs_all.append(o)
            act_all.append(ac)
            print("  %s scene %d: %d transitions  [%.0fs]"
                  % (os.path.basename(ck), s, len(o), time.time() - t0), flush=True)

    obs_all = np.concatenate(obs_all)
    act_all = np.concatenate(act_all)
    print("\ntotal transitions: %d (obs %s, act %s)"
          % (len(obs_all), obs_all.shape[1], act_all.shape[1]), flush=True)

    out = a.out or os.path.join(rundir, "planning_dataset.npz")
    np.savez_compressed(out, obs=obs_all, acts=act_all, H=a.H, K=a.K,
                        steps=a.steps, scenes=scenes)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
