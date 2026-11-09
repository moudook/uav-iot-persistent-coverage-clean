"""Evaluate the classical OR baseline on stress regimes (zero-shot).

Same controller as eval_classical_baseline.py but on stress-regime configs,
so we can answer: where does a strong classical scheduler beat/ooose to the
frozen learner?  Matches the stress protocol: 5 held-out scenes, same
regime configs.

Run:
    python scripts/eval_classical_stress.py --eval-seeds 5 --steps 500
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl", "analysis"))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import linear_sum_assignment
from uav_wireless.env import UAVIoTEnv
from stress_regimes import get_config, REGIMES
from feasibility import search_best

A_HOVER, A_PX, A_NX, A_PY, A_NY, A_CHARGE = 0, 1, 2, 3, 4, 5


def move_toward(env, i, target):
    dx = target[0] - env.p[i][0]
    dy = target[1] - env.p[i][1]
    if abs(dx) >= abs(dy):
        return A_PX if dx > 0 else (A_NX if dx < 0 else A_HOVER)
    else:
        return A_PY if dy > 0 else (A_NY if dy < 0 else A_HOVER)


def classical_eval(cfg, steps, seed):
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = UAVIoTEnv(c)
    env.reset()
    pos, ck = search_best(env, iters=80, restarts=12,
                          rng=np.random.default_rng(seed + 7919),
                          connected_only=True)
    if pos is None:
        pos = np.array([[cfg.side * (i % 4 + 0.5) / 4.0,
                         cfg.side * (i // 4 + 0.5) / 2.0] for i in range(cfg.N)])
    formation = np.asarray(pos, dtype=float)
    for _ in range(steps):
        acts = np.zeros(cfg.N, dtype=np.int64)
        active = list(env._active_idx())
        charging = set()
        for i in active:
            if env.E[i] <= env.return_threshold(i):
                charging.add(i)
        serving = [i for i in active if i not in charging]
        if serving and len(formation) >= len(serving):
            cost = np.zeros((len(serving), len(formation)))
            for si, i in enumerate(serving):
                for sj in range(len(formation)):
                    cost[si, sj] = np.linalg.norm(env.p[i] - formation[sj])
            row_ind, col_ind = linear_sum_assignment(cost)
            for si, sj in zip(row_ind, col_ind):
                i = serving[si]
                target = formation[sj]
                if np.linalg.norm(env.p[i] - target) < 2.0:
                    acts[i] = A_HOVER
                else:
                    acts[i] = move_toward(env, i, target)
        for i in charging:
            if np.min(np.linalg.norm(env.pads - env.p[i], axis=1)) <= cfg.pad_service_radius:
                acts[i] = A_CHARGE
            else:
                nearest_pad = env.pads[int(np.argmin(np.linalg.norm(env.pads - env.p[i], axis=1)))]
                acts[i] = move_toward(env, i, nearest_pad)
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    regimes = a.only or list(REGIMES.keys())
    results = {}
    t0 = time.time()
    for regime in regimes:
        cfg = get_config(regime)
        cfg.episode_len = a.steps
        cks = []
        for s in scenes:
            summ = classical_eval(cfg, a.steps, s)
            cks.append(float(summ.get("CK", 0.0)))
        results[regime] = {"per_scene": cks,
                           "mean": float(np.mean(cks)),
                           "sd": float(np.std(cks, ddof=1)) if len(cks) > 1 else 0.0}
        print("%-14s classical CK=%.4f (sd %.4f)  [%.0fs]"
              % (regime, results[regime]["mean"], results[regime]["sd"],
                 time.time() - t0), flush=True)
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "ml_impl",
                                "runs", "classical_stress.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"steps": a.steps, "scenes": scenes, "regimes": results},
                  f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
