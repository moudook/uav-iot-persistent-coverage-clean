"""Innovation 10: feasibility-aware planning (H-step lookahead at eval time).

For each decision, sample K candidate joint action vectors from the actor's
categorical distribution, roll each forward H steps in a CLONED environment
(clean state), and score by the cumulative feasibility-masked coverage (the
per-step CK of the lookahead). Pick the best candidate and execute it in the
real environment. This is model-based planning with the feasibility mask as
the constraint, using the already-trained policy as the proposal
distribution - no retraining.

Run:
    python scripts/eval_planning_v2.py --tag exp_ego_eval \
        --eval-offset 1000 --eval-seeds 20 --H 3 --K 8
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
from uav_wireless.experiments import base_config, eval_agent
from uav_wireless.mappo import MAPPO


def action_masks_safe(env):
    mask = np.stack(env.action_masks())
    for i in range(env.cfg.N):
        if mask[i].sum() == 0:
            mask[i, 0] = True   # HOVER fallback
    return mask


def rollout_score(env, agent, cand, H):
    """Clone env, execute `cand` then follow the deterministic policy for H
    steps, return cumulative per-step CK (feasibility-masked coverage)."""
    env2 = copy.deepcopy(env)
    sc = 0.0
    a = cand
    for h in range(H):
        obs2, _, r, done, _ = env2.step(a)
        summ = env2.summary()
        sc += float(summ.get("CK", 0.0))
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
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.tag)
    ckpts = sorted(glob.glob(os.path.join(rundir, "seed_*.pt")))
    if not ckpts:
        print("no checkpoints in", rundir)
        return 1
    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    print("planning H=%d K=%d on %d scenes x %d ckpts" % (a.H, a.K, len(scenes), len(ckpts)), flush=True)

    results = {}
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
        cks = []
        for s in scenes:
            cks.append(planned_eval(agent, cfg, a.steps, s, a.H, a.K))
        name = os.path.basename(ck)
        results[name] = {"mean": float(np.mean(cks)),
                         "std": float(np.std(cks, ddof=1)) if len(cks) > 1 else 0.0,
                         "per_scene": cks}
        print("%-14s planned CK=%.4f (sd %.4f)  [%.0fs]"
              % (name, results[name]["mean"], results[name]["std"], time.time() - t0), flush=True)

    means = [r["mean"] for r in results.values()]
    summary = {"tag": a.tag, "H": a.H, "K": a.K, "steps": a.steps,
               "scenes": scenes,
               "per_checkpoint": results,
               "mean_over_seeds": float(np.mean(means)),
               "sd_over_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
               "n_seeds": len(means)}
    out = a.out or os.path.join(rundir, "eval_planning.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("\n%s planning: mean over %d seeds = %.4f (sd %.4f)"
          % (a.tag, len(means), summary["mean_over_seeds"], summary["sd_over_seeds"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
