"""
Training runner with checkpointing, resumption, and a results journal.

Why this exists: the first version trained inside ad-hoc `python -c` calls, so
no run was reproducible and no result could be re-verified. Every number that
goes into the report must be traceable to a saved checkpoint plus a saved
metrics file. This module is the only sanctioned way to produce those numbers.

Usage:
    python -m uav_wireless.runner train  --steps 10000 --iters 60 --tag v2
    python -m uav_wireless.runner eval   --ckpt runs/v2/best.pt --seeds 5
    python -m uav_wireless.runner baselines --tag v2
    python -m uav_wireless.runner ablate --tag v2
    python -m uav_wireless.runner matrix --tag v2
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import time

import numpy as np
import torch

from .env import SimConfig, UAVIoTEnv
from .mappo import MAPPO
from .train import collect_rollout, baseline_fixed, baseline_greedy, baseline_reactive


# ----------------------------------------------------------------------
# Run directory / journal
# ----------------------------------------------------------------------
def run_dir(tag: str) -> str:
    d = os.path.join("runs", tag)
    os.makedirs(d, exist_ok=True)
    return d


def journal(tag: str, record: dict):
    """Append one JSON line to the run's metrics journal."""
    path = os.path.join(run_dir(tag), "journal.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_journal(tag: str):
    path = os.path.join(run_dir(tag), "journal.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
def train(
    cfg: SimConfig,
    steps: int = 10_000,
    iters: int = 60,
    tag: str = "v2",
    seed: int = 0,
    resume: bool = True,
    device: str | None = None,
    log_every: int = 5,
):
    """
    One iteration = one full episode of `steps` control steps, followed by a
    PPO update on that episode's rollout. Checkpoints are written every
    iteration so a crash costs at most one episode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = copy.deepcopy(cfg)
    cfg.episode_len = steps
    cfg.seed = seed

    d = run_dir(tag)
    ckpt_path = os.path.join(d, "last.pt")
    best_path = os.path.join(d, "best.pt")

    env = UAVIoTEnv(cfg)
    agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim,
                  state_dim=env.state_dim, device=device)

    start_it = 0
    best_ck = -1.0
    if resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        if "actor_opt" in state:
            agent.actor_opt.load_state_dict(state["actor_opt"])
            agent.critic_opt.load_state_dict(state["critic_opt"])
        elif "opt" in state:
            agent.actor_opt.load_state_dict(state["opt"])
            agent.critic_opt.load_state_dict(state["opt"])
        start_it = state["iter"] + 1
        best_ck = state.get("best_ck", -1.0)
        print(f"[resume] {ckpt_path} at iteration {start_it}")

    print(f"[train] device={device} steps/ep={steps} iters={iters} tag={tag}")
    t0 = time.time()

    for it in range(start_it, iters):
        B, summ = collect_rollout(env, agent, steps)
        losses = agent.update(B)

        # Persist every transition count so the curve is auditable.
        rec = dict(
            iter=it, tag=tag, seed=seed, steps=steps,
            wall_s=round(time.time() - t0, 1),
            **{k: float(v) for k, v in losses.items()},
            **{k: float(v) for k, v in summ.items()},
        )
        journal(tag, rec)

        ck = summ.get("CK", 0.0)
        if ck > best_ck:
            best_ck = ck
            agent.save(best_path)

        torch.save(
            dict(
                actor=agent.actor.state_dict(),
                critic=agent.critic.state_dict(),
                actor_opt=agent.actor_opt.state_dict(),
                critic_opt=agent.critic_opt.state_dict(),
                iter=it, best_ck=best_ck,
                cfg=cfg.dict(),
            ),
            ckpt_path,
        )

        if it % log_every == 0 or it == iters - 1:
            print(
                f"[{it:4d}] CK={rec.get('CK',0):.3f} AG={rec.get('AG',0):.3f} "
                f"SC={rec.get('SC',0):.3f} Crit={rec.get('Crit',0):.3f} "
                f"U={rec.get('Ubar',0):.3f} ent={rec['entropy']:.3f} "
                f"({rec['wall_s']:.0f}s)"
            )

    agent.save(os.path.join(d, "final.pt"))
    print(f"[train] done in {time.time()-t0:.0f}s -> {d}")
    return agent


# ----------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(agent, cfg: SimConfig, steps: int, seeds=range(5)):
    """Deterministic rollout over independent seeds; returns per-seed metrics."""
    out = []
    for s in seeds:
        c = copy.deepcopy(cfg)
        c.seed = s
        c.episode_len = steps
        env = UAVIoTEnv(c)
        obs, st = env.reset()
        for _ in range(steps):
            mask = np.stack(env.action_masks())
            for i in range(c.N):
                if not mask[i].any():
                    mask[i, 0] = True
            acts = agent.act(obs, mask, deterministic=True)
            obs, st, r, done, _ = env.step(acts)
            if done:
                break
        out.append(env.summary())
    return out


def summarise(runs):
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    return {
        k: dict(
            mean=float(np.mean([r[k] for r in runs])),
            std=float(np.std([r[k] for r in runs])),
            n=len(runs),
        )
        for k in keys
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _base_cfg(steps: int):
    return SimConfig(episode_len=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "baselines", "ablate", "matrix"])
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = _base_cfg(a.steps)

    if a.mode == "train":
        train(cfg, a.steps, a.iters, a.tag, a.seed, device=a.device)

    elif a.mode == "eval":
        ck = a.ckpt or os.path.join(run_dir(a.tag), "best.pt")
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Build from the checkpoint's own config: dimensions scale with N, M and
        # n_pads, so a checkpoint trained under one configuration cannot be
        # loaded into networks sized for another.
        blob = torch.load(ck, map_location="cpu")
        saved = SimConfig(**blob["cfg"]) if "cfg" in blob else cfg
        env_probe = UAVIoTEnv(dataclasses.replace(saved, episode_len=1))
        agent = MAPPO(n_agents=saved.N, obs_dim=env_probe.obs_dim,
                      state_dim=env_probe.state_dim, device=device)
        agent.load(ck)
        runs = evaluate(agent, saved, a.steps, range(a.seeds))
        s = summarise(runs)
        print(json.dumps({k: round(v["mean"], 4) for k, v in s.items()}, indent=2))
        journal(a.tag, dict(mode="eval", ckpt=ck, summary=s))

    elif a.mode == "baselines":
        for name, fn in [("Fixed", baseline_fixed),
                         ("Greedy", baseline_greedy),
                         ("Reactive", baseline_reactive)]:
            runs = []
            for s in range(a.seeds):
                c = copy.deepcopy(cfg)
                c.seed = s
                runs.append(fn(c, steps=a.steps, seed=s))
            s = summarise(runs)
            print(f"{name:10s} " + " ".join(
                f"{k}={v['mean']:.3f}" for k, v in s.items()
                if k in ("CK", "AG", "SC", "Crit", "Energy")
            ))
            journal(a.tag, dict(mode="baseline", model=name, summary=s))

    elif a.mode == "ablate":
        from .experiments import run_ablation
        run_ablation(cfg, os.path.join(run_dir(a.tag), "ablation.csv"))

    elif a.mode == "matrix":
        from .experiments import run_scenarios
        run_scenarios(cfg, os.path.join(run_dir(a.tag), "scenarios.csv"))


if __name__ == "__main__":
    main()