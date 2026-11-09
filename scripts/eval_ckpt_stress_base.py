"""Base (no planning) reference for stress regimes at a given step count,
mirroring eval_planning_stress.py's protocol so comparisons are fair.
"""
import argparse, copy, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, torch
from uav_wireless.env import UAVIoTEnv
from uav_wireless.experiments import base_config, eval_agent
from uav_wireless.mappo import MAPPO
from stress_regimes import get_config

SAME_DIM_REGIMES = {"nominal", "small_battery", "low_comm",
                    "failure", "multi_failure", "clustered"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_ego_eval")
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.tag)
    ckpts = sorted(glob.glob(os.path.join(rundir, "seed_*.pt")))
    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    regimes = a.only or ["small_battery", "clustered", "failure", "multi_failure", "low_comm", "nominal"]
    results = {}
    for regime in regimes:
        if regime not in SAME_DIM_REGIMES:
            continue
        cfg = get_config(regime)
        cfg.ego_sensors = True; cfg.n_ego = 3
        cfg.dist_shape_kappa = 0.5; cfg.use_w_penalty = True
        cfg.episode_len = a.steps
        per_seed = []
        for ck in ckpts:
            meta = json.load(open(ck.replace(".pt", ".json"), encoding="utf-8"))
            cfg.use_critical_anchor = meta.get("anchor", False)
            env = UAVIoTEnv(cfg)
            agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                          state_dim=env.state_dim, device="cpu")
            agent.load(ck)
            cks = [float(eval_agent(agent, cfg, steps=a.steps, seed=s, deterministic=False)["CK"]) for s in scenes]
            per_seed.append(float(np.mean(cks)))
        results[regime] = {"per_seed_base": per_seed, "mean": float(np.mean(per_seed)),
                           "sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
        print("%-14s base CK=%.4f (sd %.4f)" % (regime, results[regime]["mean"], results[regime]["sd"]), flush=True)
    out = a.out or os.path.join(rundir, "eval_ckpt_stress_base.json")
    json.dump({"tag": a.tag, "regimes": results, "steps": a.steps, "scenes": scenes}, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    raise SystemExit(main())