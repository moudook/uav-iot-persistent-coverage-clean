"""
Rollout collector + MAPPO training loop + rule-based baselines (Table VIII).

Baselines:
  Fixed          — hold initial formation, recharge at a fixed threshold
  Greedy         — lowest battery wins the pad, same motion rule
  Reactive       — deterministic uncovered-sensor + connectivity heuristic
  Vanilla MAPPO  — same learner with the proposed design switched off
"""

from __future__ import annotations

import os

import json
import numpy as np
import torch

from .env import (
    A_HOVER, A_PX, A_NX, A_PY, A_NY, A_CHARGE, N_ACTIONS,
    UAVIoTEnv, SimConfig,
)
from .mappo import MAPPO


                                                                        
                
                                                                        
def collect_rollout(env, agent, n_steps, deterministic=False, action_repeat=1):
    """
    Roll out one episode, recording per-agent transitions for the MAPPO update.
    Shared (team) reward is broadcast to every agent (Dec-POMDP, eq. 23).
    """
    obs, state = env.reset()
    n = env.cfg.N

    B = dict(
        obs=np.zeros((n_steps, n, env.obs_dim), np.float32),
        mask=np.zeros((n_steps, n, N_ACTIONS), bool),
        acts=np.zeros((n_steps, n), np.int64),
        logp=np.zeros((n_steps, n), np.float32),
        adv=np.zeros((n_steps, n), np.float32),
        ret=np.zeros((n_steps, n), np.float32),
        states=np.zeros((n_steps, env.state_dim), np.float32),
        values=np.zeros((n_steps, n), np.float32),
        active=np.zeros((n_steps, n), bool),
    )

    t = 0
    truncated = False
    while t < n_steps:
        mask = np.stack(env.action_masks())
        if mask.sum(axis=1).min() == 0:                                        
            for i in range(n):
                if mask[i].sum() == 0:
                    mask[i] = True
                    mask[i, A_HOVER] = True

        if agent is None:
            acts = np.zeros(n, dtype=np.int64)
            logp = np.zeros(n, np.float32)
        else:
            import torch
            acts = agent.act(obs, mask, deterministic=deterministic)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=agent.device)
            with torch.no_grad():
                lp = torch.log_softmax(agent.masked_logits(obs_t, mask_t), -1)
                logp = lp.gather(-1, torch.as_tensor(acts, device=lp.device).unsqueeze(-1)).squeeze(-1)
                logp = logp.cpu().numpy()

        import torch
        with torch.no_grad():
            v = agent.value(torch.as_tensor(state, dtype=torch.float32, device=agent.device)).cpu().numpy()
        v = np.full(n, float(v))                                         

        B["obs"][t] = obs
        B["mask"][t] = mask
        B["acts"][t] = acts
        B["logp"][t] = logp
        B["states"][t] = state
        B["values"][t] = v
        B["active"][t] = (~env.failed) & (env.mode != -1)

                                      
        r_acc = 0.0; done_acc = False
        for _ in range(action_repeat):
            obs, state, r, done, _ = env.step(acts)
            r_acc += r
            done_acc = done_acc or done
            if done: break
                                                                              
        ar = getattr(env, "last_agent_rewards", None)
        if ar is not None:
            B["adv"][t] = ar
        else:
            B["adv"][t] = r
        t += 1
        if done_acc:
            break
                                                             
     
                                                                             
                                                                              
                                                                               
                                                                             
                                                                          
                                                    
    env_terminated = bool(getattr(env, "terminated", False))
    truncated = not env_terminated
    terminal_state = state if truncated else None

                                                                        
                                                                         
                                                                          
                                                                            
                                                         
    adv = np.zeros_like(B["adv"])
    last_gae = np.zeros(n, np.float32)
                                                                  
    if truncated and agent is not None and terminal_state is not None:
        import torch
        with torch.no_grad():
            v_boot = agent.value(
                torch.as_tensor(terminal_state, dtype=torch.float32, device=agent.device)
            ).cpu().numpy()
        bootstrap_val = np.full(n, float(v_boot))
    else:
        bootstrap_val = np.zeros(n, np.float32)
    for step in reversed(range(t)):
        if step + 1 < t:
            next_val = B["values"][step + 1]
        else:
            next_val = bootstrap_val
        delta = B["adv"][step] + agent_gamma(agent) * next_val - B["values"][step]
        last_gae = delta + agent_gamma(agent) * agent_lam(agent) * last_gae
        adv[step] = last_gae
    B["adv"] = adv
                                                                           
                                                                           
                                                                         
                                                                     
                                                                       
    B = {k: v[:t] for k, v in B.items()}
    B["ret"] = B["adv"] + B["values"]
    return B, env.summary()


def agent_gamma(agent):
    return agent.gamma if agent else 0.99


def agent_lam(agent):
    return agent.lam if agent else 0.95


def agent_obs_dim(agent):
    """Input width of the agent's actor network.

    Replaced a stub that returned the literal 32. That hardcoded value matched
    neither the observation layout (which used to write only 24 of 32 slots and
    now writes exactly 30) nor any configuration other than the reference one,
    so a buffer sized from it silently disagreed with the environment.
    """
    first = next(agent.actor.parameters())
    return int(first.shape[-1])


                                                                        
          
                                                                        
def train_mappo(cfg: SimConfig, episodes=120, n_steps=400, seed=0, log_every=10,
               verbose=True, per_agent_adv_norm=False, ent_coef=0.01,
               value_norm=True, device=None, bc_path=None,
               gamma=0.99, gae_lambda=0.95, epochs=10, n_minibatch=4,
               curriculum=False, rf_at=None, lr_decay=1.0,
               attn=False, action_repeat=1,
               ckpt_dir=None, ckpt_every=25):
    """Train the proposed controller.

    THE DEVICE IS A PROTOCOL CHOICE, NOT A RESOURCE ACCIDENT.  The
    floating-point kernels differ between the CPU and the GPU, so the same
    seed trained on the two devices gives different weights (the difference
    is small, about 6e-4 after one episode, and it compounds through the
    updates).  A device chosen by "is there room on the GPU right now"
    therefore makes the result depend on the machine's momentary load, and
    two runs of the same configuration that happen to land on different
    devices are not the same experiment.  The device is consequently taken
    from the environment variable UAV_DEVICE and DEFAULTS TO THE CPU; the
    value "auto" restores the old load-dependent behaviour, and "cuda"
    asks for the GPU explicitly.

    The choice is not a matter of taste: it is measured, and the
    measurement is in analysis/device_sensitivity.json.  On this workload
    the GPU is about 1.26x SLOWER than the CPU, because the simulator is
    a sequential Python loop that is 85% of the run, while the network is
    8% of it -- and on the GPU even that 8% is nearly four times slower
    in absolute terms, since the kernels are tiny and the launch
    overhead dominates.  There is nothing for the GPU to accelerate
    here.  (The ratio is the number to quote: the absolute cost per step
    depends on what else the machine is running.)
    """
    torch.manual_seed(seed)
                                                                          
                                                                            
                                                                       
                                                                         
                                                                       
                                                  
    try:
        torch.set_num_threads(1)
    except Exception:                                                         
        pass
    env = UAVIoTEnv(cfg)
    if device is None:
        device = os.environ.get("UAV_DEVICE", "cpu")
    if device == "auto":
                                                                           
                                                                          
                                                                       
                                                                           
                                                                            
                                                                         
                                                                      
        device = "cpu"
        if torch.cuda.is_available():
            try:
                free, _total = torch.cuda.mem_get_info()
                if free > 1_500_000_000:              
                    device = "cuda"
            except Exception:
                device = "cpu"
    agent = MAPPO(n_agents=cfg.N, obs_dim=env.obs_dim, state_dim=env.state_dim,
                  per_agent_adv_norm=per_agent_adv_norm, ent_coef=ent_coef,
                  value_norm=value_norm, device=device, clip=0.2, lr=3e-4,
                  gamma=gamma, gae_lambda=gae_lambda, attn=attn)
    if bc_path:
        if os.path.exists(bc_path):
            try:
                sd = torch.load(bc_path, map_location=agent.device, weights_only=True)
                agent.actor.load_state_dict(sd)
                if verbose:
                    print(f"[train] loaded BC actor from {bc_path}")
            except Exception as e:
                if verbose:
                    print(f"[train] BC load failed ({e}); training from scratch")
    hist = []
    base_bmax = cfg.B_max
    for ep in range(episodes):
                                                                         
                                                                           
                                                                            
                                 
        if curriculum:
            frac = min(1.0, ep / max(1, episodes // 2))
            mult = 2.0 - frac              
            env.cfg.B_max = base_bmax * mult
                                                                        
                                                      
        if rf_at is not None and ep == rf_at:
            for opt in (agent.actor_opt, agent.critic_opt):
                opt.param_groups[0]["lr"] *= lr_decay
                opt.state.clear()
            if verbose:
                print(f"[train] R&F at ep {ep}: lr x{lr_decay}")
        B, summ = collect_rollout(env, agent, n_steps, action_repeat=action_repeat)
        losses = agent.update(B, epochs=epochs, n_minibatch=n_minibatch)
        rec = dict(episode=ep, mean_reward=float(B["adv"][:, 0].mean()), **losses)
        rec.update({k: float(v) for k, v in summ.items()})
                                                                       
                                                
        rec["device"] = str(agent.device)
        hist.append(rec)
                                                       
        if ckpt_dir and (ep % ckpt_every == 0 or ep == episodes - 1):
            os.makedirs(ckpt_dir, exist_ok=True)
            agent.save(os.path.join(ckpt_dir, "latest.pt"))
            with open(os.path.join(ckpt_dir, "partial.json"), "w") as _f:
                json.dump({"ep": ep, "episodes": episodes,
                           "train_ck": float(hist[-1].get("CK", 0.0)),
                           "hist": hist}, _f)
        if verbose and (ep % log_every == 0 or ep == episodes - 1):
            print(
                f"[MAPPO ep {ep:3d}] R={rec['mean_reward']:+.4f} "
                f"CK={rec.get('CK',0):.3f} AG={rec.get('AG',0):.3f} "
                f"Crit={rec.get('Crit',0):.3f} ent={rec['entropy']:.3f}"
            )
    return agent, hist


                                                                        
                                   
                                                                        
def _seeded(cfg: SimConfig, seed: int) -> SimConfig:
    """Copy of cfg with the episode seed applied.

    The seed selects the random scene (sensor layout, pad placement, initial
    positions and the shadowing draw).  Every baseline must apply it, or a
    "five-seed" comparison degenerates into the same episode run five times
    while the learned policy is evaluated on five different scenes.
    """
    import copy
    c = copy.deepcopy(cfg)
    c.seed = int(seed)
    return c


def _move_toward(env, i, target):
    c = env.cfg
    d = target - env.p[i]
    if np.linalg.norm(d) < 1e-6:
        return A_HOVER
    step = c.v_max * c.dt
                                                                
    if abs(d[0]) >= abs(d[1]):
        return A_PX if d[0] > 0 else A_NX
    return A_PY if d[1] > 0 else A_NY


def _uncovered_target(env):
    ncov = env.coverage_counts()
    under = np.where(ncov < env.cfg.K)[0]
    if len(under):
        return env.sensors[under[np.argmin(np.linalg.norm(env.sensors[under] - env.p.mean(0), axis=1))]]
    return env.pads[0]


def baseline_fixed(cfg: SimConfig, steps=1000, seed=0):
    """Fixed formation with threshold recharge.

    The `seed` argument selects the episode's random scene (sensor layout,
    pad placement, initial positions).  It MUST be applied to the config:
    accepting the argument and ignoring it would make every "seed" in a
    multi-seed comparison the same episode, which silently turns a
    five-seed comparison into one repeated run.
    """
    cfg = _seeded(cfg, seed)
    env = UAVIoTEnv(cfg)
    obs, st = env.reset()
    for _ in range(steps):
        acts = np.zeros(cfg.N, dtype=np.int64)
        for i in env._active_idx():
            if env.E[i] <= env.return_threshold(i):
                                                                     
                if np.min(np.linalg.norm(env.pads - env.p[i], axis=1)) <= env.cfg.pad_service_radius:
                    acts[i] = A_CHARGE
                else:
                    acts[i] = _move_toward(env, i, env.pads[int(np.argmin(np.linalg.norm(env.pads - env.p[i], axis=1)))])
            else:
                acts[i] = A_HOVER
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()


def baseline_greedy(cfg: SimConfig, steps=1000, seed=0):
    """Greedy lowest-battery recharge with the same motion rule."""
    cfg = _seeded(cfg, seed)
    env = UAVIoTEnv(cfg)
    obs, st = env.reset()
    for _ in range(steps):
        acts = np.zeros(cfg.N, dtype=np.int64)
        for i in env._active_idx():
            if env.E[i] <= env.return_threshold(i):
                if np.min(np.linalg.norm(env.pads - env.p[i], axis=1)) <= env.cfg.pad_service_radius:
                    acts[i] = A_CHARGE
                else:
                    acts[i] = _move_toward(env, i, env.pads[int(np.argmin(np.linalg.norm(env.pads - env.p[i], axis=1)))])
            else:
                acts[i] = _move_toward(env, i, _uncovered_target(env))
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()


def baseline_reactive_local(cfg: SimConfig, steps=1000, seed=0):
    """The same reactive heuristic, WITHOUT the privileged global state.

    `baseline_reactive` assigns the swarm to the under-covered sensors in a
    global round-robin: it reads the coverage counts of EVERY sensor, so
    it knows which sensors are under-covered and which UAV is assigned to
    which.  That is an unbounded broadcast of the coverage state, and the
    learned policy receives a fixed-size summary of the same information.
    Part of the gap between them is therefore a difference of information
    and not of control.

    This variant removes most of that privilege while keeping the rule
    hand-designed and the motion logic identical.  Each UAV uses ONLY:

      * its own position, battery and mode;
      * the STATIC sensor layout and the static pad layout, which are
        infrastructure rather than coverage state and which a deployed
        UAV can be told once;
      * the coverage counts of the sensors IT is close enough to judge for
        itself -- a radius of twice R_cov, which is what a UAV can
        observe locally.

    It does not use the global under-covered list, and it does not use an
    assignment: each UAV moves toward the nearest sensor that IT judges
    to be under-covered.  The resulting contention between UAVs that want
    the same sensor is exactly the difficulty the learned policy faces,
    and it is the point of the variant.
    """
    cfg = _seeded(cfg, seed)
    env = UAVIoTEnv(cfg)
    obs, st = env.reset()
    local_radius = 2.0 * cfg.R_cov
    for _ in range(steps):
        acts = np.zeros(cfg.N, dtype=np.int64)
        ncov = env.coverage_counts()
        for i in env._active_idx():
            if env.E[i] <= env.return_threshold(i):
                if np.min(np.linalg.norm(env.pads - env.p[i], axis=1)) <= cfg.pad_service_radius:
                    acts[i] = A_CHARGE
                else:
                    acts[i] = _move_toward(env, i, env.pads[int(np.argmin(np.linalg.norm(env.pads - env.p[i], axis=1)))])
                continue
                                                       
            d = np.linalg.norm(env.sensors - env.p[i], axis=1)
            local = np.nonzero(d <= local_radius)[0]
            under_local = local[ncov[local] < cfg.K]
            if len(under_local) == 0:
                acts[i] = A_HOVER
                continue
            tgt = env.sensors[under_local[int(np.argmin(d[under_local]))]]
            acts[i] = _move_toward(env, i, tgt)
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()


def baseline_reactive(cfg: SimConfig, steps=1000, seed=0):
    """Deterministic coverage + connectivity heuristic (no learning)."""
    cfg = _seeded(cfg, seed)
    env = UAVIoTEnv(cfg)
    obs, st = env.reset()
    for _ in range(steps):
        acts = np.zeros(cfg.N, dtype=np.int64)
        ncov = env.coverage_counts()
        under = np.where(ncov < cfg.K)[0]
        for k, i in enumerate(env._active_idx()):
            if env.E[i] <= env.return_threshold(i):
                if np.min(np.linalg.norm(env.pads - env.p[i], axis=1)) <= cfg.pad_service_radius:
                    acts[i] = A_CHARGE
                else:
                    acts[i] = _move_toward(env, i, env.pads[int(np.argmin(np.linalg.norm(env.pads - env.p[i], axis=1)))])
                continue
            if len(under) == 0:
                acts[i] = A_HOVER
                continue
                                                                                 
            tgt = env.sensors[under[k % len(under)]]
            acts[i] = _move_toward(env, i, tgt)
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()

