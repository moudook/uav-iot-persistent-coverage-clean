"""Feasibility-aware planning on stress regimes (zero-shot, test-time).

Same planner as eval_planning_v2.py but evaluated on the stress-regime
configs (small_battery, clustered/spread_start, failure, ...), using the
frozen nominal checkpoints of a tag. Reuses the regime handling from
eval_stress.py.

Run:
    python scripts/eval_planning_stress.py --tag exp_ego_eval \
        --only small_battery clustered failure --H 2 --K 4 --steps 500
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from uav_wireless.env import UAVIoTEnv
from uav_wireless.experiments import base_config
from uav_wireless.mappo import MAPPO
from stress_regimes import get_config, REGIMES

# regimes that keep the observation/state dims unchanged (zero-shot capable)
SAME_DIM_REGIMES = {"nominal", "small_battery", "low_comm",
                    "failure", "multi_failure", "clustered"}


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


def planned_eval(agent, cfg, steps, seed, H, K):
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = UAVIoTEnv(c)
    obs, _ = env.reset()
    rng = np.random.default_rng(seed * 7919 + 13)
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
        obs, _, r, done, _ = env.step(best_acts)
        n += 1
        if done:
            break
    return float(env.summary().get("CK", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_ego_eval")
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.tag)
    ckpts = sorted(glob.glob(os.path.join(rundir, "seed_*.pt")))
    if not ckpts:
        print("no checkpoints in", rundir)
        return 1
    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    regimes = a.only or list(REGIMES.keys())
    print("planning stress H=%d K=%d steps=%d on %d scenes x %d ckpts"
          % (a.H, a.K, a.steps, len(scenes), len(ckpts)), flush=True)

    results = {}
    for regime in regimes:
        if regime not in SAME_DIM_REGIMES:
            print("skip (dims change):", regime, flush=True)
            continue
        print("\n=== regime:", regime, "===", flush=True)
        cfg = get_config(regime)
        cfg.ego_sensors = True
        cfg.n_ego = 3
        cfg.dist_shape_kappa = 0.5
        cfg.use_w_penalty = True
        cfg.episode_len = a.steps
        per_seed = []
        t0 = time.time()
        for ck in ckpts:
            meta = json.load(open(ck.replace(".pt", ".json"), encoding="utf-8"))
            cfg.use_critical_anchor = meta.get("anchor", False)
            env = UAVIoTEnv(cfg)
            agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                          state_dim=env.state_dim, device="cpu")
            agent.load(ck)
            cks = []
            for s in scenes:
                cks.append(planned_eval(agent, cfg, a.steps, s, a.H, a.K))
            per_seed.append(float(np.mean(cks)))
            print("  %-14s planned CK=%.4f  [%.0fs]"
                  % (os.path.basename(ck), per_seed[-1], time.time() - t0), flush=True)
        results[regime] = {"per_seed_planning": per_seed,
                           "mean": float(np.mean(per_seed)),
                           "sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                           "scenes": scenes, "H": a.H, "K": a.K}
        print("  regime mean=%.4f (sd %.4f)" % (results[regime]["mean"], results[regime]["sd"]), flush=True)

    out = a.out or os.path.join(rundir, "eval_planning_stress.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tag": a.tag, "regimes": results, "steps": a.steps,
                   "H": a.H, "K": a.K, "scenes": scenes}, f, indent=1)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())