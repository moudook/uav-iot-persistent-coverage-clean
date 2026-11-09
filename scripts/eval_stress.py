"""Zero-shot stress evaluation of the corrected learner.

The learner is trained on the nominal configuration and then rolled out,
without retraining, on each stress regime of scripts/stress_regimes.py.  The
rules are measured on the same scenes.  This is the experiment that decides
whether the thesis ``certifying where learning does not help'' is supported
on stress or only on the base configuration.

Writes ml_impl/runs/stress_eval/stress_eval.json
"""
import argparse
import json
import os
import sys
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from uav_wireless.env import UAVIoTEnv
from uav_wireless.experiments import base_config, eval_agent
from uav_wireless.mappo import MAPPO
from uav_wireless.train import (baseline_fixed, baseline_greedy,
                                baseline_reactive, baseline_reactive_local)
from stress_regimes import get_config, REGIMES

# the learner checkpoints were trained in this configuration (ego
# bearings + distance shaping); this must be applied on top of every
# regime override when the learner is evaluated.
LEARNER_ANCHOR = dict(ego_sensors=True, n_ego=3,
                      dist_shape_kappa=0.5, use_w_penalty=True,
                      use_critical_anchor=False)

def learner_anchor_from_meta(ckpt_list):
    """Read the learner config flags from the first checkpoint's training
    metadata, so stress evaluation uses the same observation the policy
    was trained with (incl. the critical-coverage anchor feature)."""
    import json as _j
    anchor = dict(LEARNER_ANCHOR)
    if ckpt_list:
        try:
            meta = _j.load(open(ckpt_list[0].replace(".pt", ".json"), encoding="utf-8"))
            anchor["ego_sensors"] = meta.get("ego", True)
            anchor["n_ego"] = meta.get("n_ego", 3)
            anchor["dist_shape_kappa"] = meta.get("shape", 0.5)
            anchor["use_w_penalty"] = meta.get("use_w", True)
            anchor["use_critical_anchor"] = meta.get("anchor", False)
        except Exception:
            pass
    return anchor

# Regimes that change the size of the observation or the state cannot
# load a checkpoint trained on the nominal architecture: the actor and
# critic input dims are fixed by N, M and n_pads.  Zero-shot
# evaluation is therefore only defined on the regimes below.
SAME_DIM_REGIMES = {"nominal", "small_battery", "low_comm",
                    "failure", "multi_failure", "clustered"}

RULES_ALL = [("Fixed", baseline_fixed), ("Greedy", baseline_greedy),
            ("Reactive", baseline_reactive), ("Reactive-local", baseline_reactive_local)]
RULES = None  # set from --rules


def eval_learner_on_regime(ckpt_list, cfg, scenes, steps):
    per_seed = []
    for ck in ckpt_list:
        import json as _j
        meta = _j.load(open(ck.replace(".pt", ".json"), encoding="utf-8"))
        # build env to get dims
        env = UAVIoTEnv(cfg)
        ag = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                   state_dim=env.state_dim, device="cpu")
        ag.load(ck)
        cks = []
        with torch.no_grad():
            for s in scenes:
                r = eval_agent(ag, cfg, steps=steps, seed=s, deterministic=False)
                cks.append(float(r["CK"]))
        per_seed.append(float(np.mean(cks)))
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp_ego_eval")
    ap.add_argument("--eval-offset", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--only", nargs="+", default=None,
                    help="only these regimes")
    ap.add_argument("--rules", nargs="+",
                    default=["Fixed", "Reactive-local"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rundir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", a.tag)
    ckpts = sorted(__import__("glob").glob(os.path.join(rundir, "seed_*.pt")))
    scenes = list(range(a.eval_offset, a.eval_offset + a.eval_seeds))
    results = {}
    global RULES
    RULES = [x for x in RULES_ALL if x[0] in a.rules]
    regimes = a.only or list(REGIMES.keys())
    for regime in regimes:
        print("\n=== regime:", regime, "===", flush=True)
        rule_cfg = get_config(regime)
        rule_cfg.episode_len = a.steps
        learner_cfg = get_config(regime)
        _anchor_flags = learner_anchor_from_meta(ckpts)
        for _k, _v in _anchor_flags.items():
            setattr(learner_cfg, _k, _v)
        learner_cfg.episode_len = a.steps
        # rules
        rule_summary = {}
        for name, fn in RULES:
            cks = [float(fn(rule_cfg, steps=a.steps, seed=s)["CK"]) for s in scenes]
            rule_summary[name] = {"CK": float(np.mean(cks)),
                                  "sd": float(np.std(cks, ddof=1)),
                                  "per_scene": cks}
        best_rule = max(rule_summary.values(), key=lambda v: v["CK"])
        best_name = max(rule_summary, key=lambda k: rule_summary[k]["CK"])
        # learner (its own anchor config on top of the regime); only where
        # the architecture dims are unchanged
        if regime in SAME_DIM_REGIMES:
            learner_per_seed = eval_learner_on_regime(ckpts, learner_cfg, scenes, a.steps)
        else:
            learner_per_seed = []
        learner_mean = (float(np.mean(learner_per_seed))
                        if learner_per_seed else float("nan"))
        learner_sd = (float(np.std(learner_per_seed, ddof=1))
                      if len(learner_per_seed) > 1 else 0.0)
        headroom = 1.0 - rule_summary[best_name]["CK"]
        results[regime] = {
            "rules": rule_summary,
            "best_rule": best_name,
            "best_rule_ck": rule_summary[best_name]["CK"],
            "learner_mean": learner_mean,
            "learner_sd": learner_sd,
            "learner_per_seed": learner_per_seed,
            "headroom": headroom,
            "learner_minus_rule": (learner_mean - rule_summary[best_name]["CK"]
                                   if learner_per_seed else None),
        }
        print("  best rule %-16s CK=%.4f  headroom=%.4f" % (best_name, rule_summary[best_name]["CK"], headroom))
        print("  learner (zero-shot) CK=%.4f (sd %.4f)" % (learner_mean, learner_sd))
        print("  learner - rule = %+.4f" % (learner_mean - rule_summary[best_name]["CK"]))

    outdir = os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", "stress_eval")
    os.makedirs(outdir, exist_ok=True)
    out = a.out or os.path.join(outdir, "stress_eval.json")
    # merge with an existing artifact (the --only runs update single regimes)
    merged = {}
    if os.path.exists(out):
        try:
            merged = json.load(open(out, encoding="utf-8")).get("regimes", {})
        except Exception:
            merged = {}
    merged.update(results)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"regimes": merged, "scenes": scenes, "steps": a.steps,
                   "tag": a.tag, "n_seeds": len(ckpts)}, f, indent=1)
    print("\nwrote", out)
    # summary table
    print("\n%-16s %-18s %8s %8s %8s" % ("regime", "best rule", "rule CK", "learner", "headroom"))
    for r, v in results.items():
        print("%-16s %-18s %8.4f %8.4f %8.4f" % (r, v["best_rule"], v["best_rule_ck"],
              v["learner_mean"], v["headroom"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
