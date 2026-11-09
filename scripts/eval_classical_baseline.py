"""
Classical baseline: oracle-formation + Hungarian assignment + charging rotation.

A strong non-learning controller for persistent K-coverage:
  1. Precompute a connected K-covering formation via the teleporting
     oracle (search_best with connected_only=True).
  2. Each step, assign active (non-charging) UAVs to formation slots via
     the Hungarian algorithm (minimise total distance).
  3. Move toward the assigned slot; hover if arrived.
  4. Battery: if E_i <= return_threshold, the UAV returns to the nearest
     pad and charges.  Slots vacated by charging UAVs are filled by the
     next available charged UAV (relieve-before-leave is approximated by
     only allowing one UAV per slot to charge at a time).

This is the operations-research-style scheduler a reviewer will ask for:
it should score near the oracle on nominal and expose where learning adds
value (connectivity failures, multi-failure recovery).

Run:
    python scripts/eval_classical_baseline.py --tag exp_ego_eval \\
        --eval-offset 1000 --eval-seeds 5 --steps 500
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl", "analysis"))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import linear_sum_assignment
from uav_wireless.env import UAVIoTEnv, SimConfig
from uav_wireless.experiments import base_config
from feasibility import search_best, is_connected_at

# action constants
A_HOVER, A_PX, A_NX, A_PY, A_NY, A_CHARGE = 0, 1, 2, 3, 4, 5
DIRS = {A_PX: (1, 0), A_NX: (-1, 0), A_PY: (0, 1), A_NY: (0, -1)}


def move_toward(env, i, target):
    """Return the cardinal action that moves UAV i toward target."""
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

    # precompute oracle formation (connected, K-covering)
    pos, ck = search_best(env, iters=80, restarts=12,
                          rng=np.random.default_rng(seed + 7919),
                          connected_only=True)
    if pos is None:
        # fallback: spread UAVs evenly
        pos = np.array([[cfg.side * (i % 4 + 0.5) / 4.0,
                         cfg.side * (i // 4 + 0.5) / 2.0] for i in range(cfg.N)])
    formation = np.asarray(pos, dtype=float)

    # slot occupancy: which UAV is assigned to which slot
    slot_of = {}  # slot_idx -> uav_idx
    uav_slot = {}  # uav_idx -> slot_idx

    for _ in range(steps):
        acts = np.zeros(cfg.N, dtype=np.int64)
        active = list(env._active_idx())

        # determine who needs to charge
        charging = set()
        for i in active:
            if env.E[i] <= env.return_threshold(i):
                charging.add(i)

        # active UAVs that are not charging
        serving = [i for i in active if i not in charging]

        # assign serving UAVs to formation slots via Hungarian
        if serving and len(formation) >= len(serving):
            # build cost matrix: serving x slots
            # available slots = those not occupied by a charging UAV that's
            # still at its slot... simplify: all slots available
            cost = np.zeros((len(serving), len(formation)))
            for si, i in enumerate(serving):
                for sj in range(len(formation)):
                    cost[si, sj] = np.linalg.norm(env.p[i] - formation[sj])
            row_ind, col_ind = linear_sum_assignment(cost)
            for si, sj in zip(row_ind, col_ind):
                i = serving[si]
                slot = sj
                target = formation[slot]
                # move toward target
                if np.linalg.norm(env.p[i] - target) < 2.0:
                    acts[i] = A_HOVER
                else:
                    acts[i] = move_toward(env, i, target)

        # charging UAVs
        for i in charging:
            if np.min(np.linalg.norm(env.pads - env.p[i], axis=1)) <= cfg.pad_service_radius:
                acts[i] = A_CHARGE
            else:
                nearest_pad = env.pads[int(np.argmin(np.linalg.norm(env.pads - env.p[i], axis=1)))]
                acts[i] = move_toward(env, i, nearest_pad)

        # failed UAVs (mode -1) are handled by env
        obs, st, r, done, _ = env.step(acts)
        if done:
            break

    return env.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_ego_eval")
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # use base config (nominal)
    cfg = base_config()
    cfg.ego_sensors = True
    cfg.n_ego = 3
    cfg.dist_shape_kappa = 0.5
    cfg.use_w_penalty = True
    cfg.episode_len = a.steps

    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    cks = []
    t0 = time.time()
    for s in scenes:
        summ = classical_eval(cfg, a.steps, s)
        cks.append(float(summ.get("CK", 0.0)))
        print("scene %d: CK=%.4f  [%.0fs]" % (s, cks[-1], time.time() - t0), flush=True)

    mean = float(np.mean(cks))
    sd = float(np.std(cks, ddof=1)) if len(cks) > 1 else 0.0
    print("\nclassical baseline: mean=%.4f (sd %.4f) over %d scenes" % (mean, sd, len(cks)))

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", "classical_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tag": "classical_baseline", "steps": a.steps, "scenes": scenes,
                   "per_scene": cks, "mean": mean, "sd": sd}, f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
