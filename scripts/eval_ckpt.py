"""Evaluate saved checkpoints on a controlled set of scenes.

Evaluation is decoupled from training on purpose.  A policy is trained once and
then measured as many times as the protocol needs, on scene sets chosen after
the fact, without retraining: the headline number of the paper is the frozen
policy's mean over a set of scenes, and a number that required retraining to
recompute is a number that gets frozen by accident.

Scene selection
---------------
A scene is a function of the environment seed.  Training consumes seeds
0, 1, 2, ... so the evaluation scenes are drawn from an offset range
(default 1000..) that training never touches: no evaluation scene is a
training scene, which is what "held out" has to mean if it is to mean
anything.

Run:
    python scripts/eval_ckpt.py --tag exp_ego_eval --eval-offset 1000 \
        --eval-seeds 20
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from uav_wireless.env import UAVIoTEnv
from uav_wireless.experiments import base_config, eval_agent
from uav_wireless.mappo import MAPPO


def build_agent(cfg, ckpt):
    env = UAVIoTEnv(cfg)
    agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                  state_dim=env.state_dim, device="cpu")
    agent.load(ckpt)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs",
                          a.tag)
    ckpts = sorted(glob.glob(os.path.join(rundir, "seed_*.pt")))
    if not ckpts:
        print("no checkpoints in", rundir)
        return 1

    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    results = {}
    for ck in ckpts:
        # the configuration is recovered from the training record, so that a
        # checkpoint is always evaluated with the environment it was trained
        # in and never with the default one
        meta_path = ck.replace(".pt", ".json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg = base_config()
        cfg.ego_sensors = meta.get("ego", False)
        cfg.n_ego = meta.get("n_ego", 3)
        cfg.dist_shape_kappa = meta.get("shape", 0.0)
        cfg.use_w_penalty = meta.get("use_w", True)
        cfg.credit_mode = meta.get("credit_mode", "team")
        cfg.randomize_layout = meta.get("randomize_layout", False)
        cfg.use_critical_anchor = meta.get("anchor", False)
        steps = a.steps or meta.get("n_steps", 2000)
        cfg.episode_len = steps

        agent = build_agent(cfg, ck)
        per, det = [], []
        with torch.no_grad():
            for s in scenes:
                r = eval_agent(agent, cfg, steps=steps, seed=s,
                               deterministic=False)
                per.append(float(r["CK"]))
                det.append(float(eval_agent(agent, cfg, steps=steps, seed=s,
                                            deterministic=True)["CK"]))
        name = os.path.basename(ck)
        results[name] = {
            "scenes": scenes, "per_scene_CK": per,
            "mean": float(np.mean(per)),
            "std": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
            "det_mean": float(np.mean(det)),
        }
        print("%-14s CK=%.4f (sd over scenes %.4f)  argmax %.4f"
              % (name, results[name]["mean"], results[name]["std"],
                 results[name]["det_mean"]), flush=True)

    means = [r["mean"] for r in results.values()]
    summary = {
        "tag": a.tag, "eval_offset": a.eval_offset,
        "eval_seeds": a.eval_seeds, "steps": a.steps or None,
        "per_checkpoint": results,
        "mean_over_seeds": float(np.mean(means)),
        "sd_over_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        "n_seeds": len(means),
    }
    out = a.out or os.path.join(rundir, "eval_heldout.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("\n%s: mean over %d seeds = %.4f (sd %.4f)"
          % (a.tag, len(means), summary["mean_over_seeds"],
             summary["sd_over_seeds"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
