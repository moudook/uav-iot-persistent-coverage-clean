"""
Learned baselines that are not the proposed method.

The paper's comparison table names a value-based multi-agent method, a
QMIX-style persistent-coverage method, and a PPO-family method that uses
action masking.  The first and the third are implemented here so that the
comparison table is not populated with methods whose numbers do not exist.
They are implementations of the METHOD FAMILIES, not reproductions of the
cited papers, and they are labelled as such in the report.

    Independent DQN (ID-DQN)
        A value-based multi-agent baseline: one Q-network per agent, each
        trained on the team reward through its own experience, epsilon-greedy
        exploration, no centralised critic and no parameter sharing.  This is
        the standard independent-learning baseline of the value-based family
        and is the slot the table's MAD-DDQN entry refers to.

    Masked PPO, local critic (PPO-Local)
        A PPO-family baseline that keeps the single ingredient this paper
        argues for -- the action mask -- but drops the other two: no
        parameter sharing and no centralised critic, so each agent has its
        own actor and its own critic on its own observation.  It is the
        slot the table's MAHPPO-AM entry refers to.

Both use exactly the same simulator, the same observation, the same action
mask and the same team reward as the proposed method, so the only things
that differ are the learning machinery.  Both are trained with the same
episode budget and evaluated on the same seeds.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_wireless.env import N_ACTIONS, UAVIoTEnv, SimConfig


# ----------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_actions=N_ACTIONS):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


# ----------------------------------------------------------------------
# Independent DQN
# ----------------------------------------------------------------------
class IDDQN:
    """One Q-network per agent; independent learning on the team reward."""

    def __init__(self, obs_dim, n_agents, hidden=128, lr=1e-3, gamma=0.99,
                 buffer=10_000, batch=128, double=True, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.buffer_size = buffer
        self.batch = batch
        self.double = double
        self.n_agents = n_agents
        self.q = QNet(obs_dim, hidden).to(device)
        self.q_target = QNet(obs_dim, hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = []          # (obs[N,O], act[N], rew, obs2[N,O], done)

    def act(self, obs, mask, deterministic=False, eps=0.0):
        """Greedy action per agent, with epsilon-greedy exploration."""
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                            device=self.device)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool,
                            device=self.device)
        with torch.no_grad():
            q = self.q(o)
        q = q.masked_fill(~m, -1e30)
        greedy = q.argmax(dim=-1).cpu().numpy()
        if deterministic or eps <= 0:
            return greedy
        rnd = np.random.randint(0, N_ACTIONS, size=len(greedy))
        pick = np.random.rand(len(greedy)) < eps
        acts = np.where(pick, rnd, greedy)
        # an exploratory action must still be feasible
        for i, a in enumerate(acts):
            if not mask[i][a]:
                acts[i] = greedy[i]
        return acts.astype(np.int64)

    def observe(self, obs, acts, rew, obs2, done):
        self.replay.append((np.asarray(obs, dtype=np.float32),
                            np.asarray(acts, dtype=np.int64),
                            float(rew),
                            np.asarray(obs2, dtype=np.float32),
                            bool(done)))
        if len(self.replay) > self.buffer_size:
            self.replay.pop(0)

    def update(self):
        if len(self.replay) < self.batch:
            return {}
        idx = np.random.randint(0, len(self.replay), self.batch)
        obs = torch.as_tensor(np.stack([self.replay[i][0] for i in idx]),
                              dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(np.stack([self.replay[i][1] for i in idx]),
                               dtype=torch.long, device=self.device)
        rew = torch.as_tensor(np.array([self.replay[i][2] for i in idx]),
                              dtype=torch.float32, device=self.device)
        obs2 = torch.as_tensor(np.stack([self.replay[i][3] for i in idx]),
                               dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.array([self.replay[i][4] for i in idx]),
                               dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_next = self.q_target(obs2)
            if self.double:
                q_online = self.q(obs2)
                a_star = q_online.argmax(dim=-1, keepdim=True)
                q_next = q_next.gather(-1, a_star).squeeze(-1)     # (B, N)
            else:
                q_next = q_next.max(dim=-1).values                 # (B, N)
            # the team reward and the done flag are per time step, shape
            # (B,); they have to be broadcast across the agents
            target = (rew[:, None, None]
                      + self.gamma * (1.0 - done)[:, None, None]
                      * q_next.unsqueeze(-1))                      # (B, N, 1)
        q = self.q(obs).gather(-1, acts.unsqueeze(-1)).squeeze(-1)  # (B, N)
        loss = F.smooth_l1_loss(q, target.squeeze(-1))
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        return dict(loss=float(loss))

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())


def train_id_dqn(cfg: SimConfig, episodes=20, steps=None, seed=0,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 log_every=5, verbose=True):
    """Train the independent-DQN baseline and return (agent, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    steps = steps or cfg.episode_len
    env = UAVIoTEnv(cfg)
    agent = IDDQN(env.obs_dim, cfg.N)
    hist = []
    eps = eps_start
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        for t in range(steps):
            mask = np.stack(env.action_masks())
            for i in range(cfg.N):
                if mask[i].sum() == 0:
                    mask[i, 0] = True
            acts = agent.act(obs, mask, eps=eps)
            obs2, st, r, done, _ = env.step(acts)
            agent.observe(obs, acts, r, obs2, done)
            obs = obs2
            ep_r += r
            if done:
                break
        for _ in range(4):                      # a few gradient steps per step
            agent.update()
        agent.sync_target()
        eps = max(eps_end, eps * eps_decay)
        s = env.summary()
        hist.append(dict(episode=ep, reward=ep_r, eps=eps, **s))
        if verbose and (ep % max(1, log_every) == 0 or ep == episodes - 1):
            print(f"[ID-DQN ep {ep:3d}] R={ep_r:8.2f} CK={s.get('CK', 0):.3f} "
                  f"AG={s.get('AG', 0):.3f} eps={eps:.3f}", flush=True)
    return agent, hist


# ----------------------------------------------------------------------
# Masked PPO with a local critic
# ----------------------------------------------------------------------
class PPOLocal:
    """Per-agent actor and per-agent local critic; no parameter sharing, no
    centralised critic.  Keeps the action mask."""

    def __init__(self, obs_dim, n_agents, hidden=128, lr=3e-4, gamma=0.99,
                 lam=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_agents = n_agents
        self.nets = [ActorCritic(obs_dim, hidden).to(device)
                      for _ in range(n_agents)]
        self.opt = torch.optim.Adam(
            [p for net in self.nets for p in net.parameters()], lr=lr)

    def act(self, obs, mask, deterministic=False):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                            device=self.device)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool,
                            device=self.device)
        acts, logps = [], []
        with torch.no_grad():
            for i, net in enumerate(self.nets):
                logits = net.actor(o[i])
                logits = logits.masked_fill(~m[i], -1e30)
                dist = torch.distributions.Categorical(logits=logits)
                a = (torch.argmax(logits) if deterministic
                     else dist.sample())
                acts.append(int(a))
                logps.append(float(dist.log_prob(a)))
        return np.array(acts, dtype=np.int64), np.array(logps, dtype=np.float32)

    def update(self, batch, adv, ret, old_logp):
        """One PPO update.  The advantage is the SAME for every agent, as it
        is for the proposed method: the team reward is broadcast."""
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32,
                              device=self.device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.bool,
                              device=self.device)
        acts = torch.as_tensor(batch["acts"], dtype=torch.long,
                               device=self.device)
        old_logp = torch.as_tensor(old_logp, dtype=torch.float32,
                                   device=self.device)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_loss = 0.0
        critic_loss = 0.0
        entropy = 0.0
        n = self.n_agents
        for i in range(n):
            net = self.nets[i]
            logits = net.actor(obs[:, i])
            logits = logits.masked_fill(~mask[:, i], -1e30)
            logp = torch.log_softmax(logits, dim=-1)
            logp_a = logp.gather(-1, acts[:, i].unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(logp_a - old_logp[:, i])
            s1 = ratio * adv[:, i]
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[:, i]
            actor_loss = actor_loss - torch.min(s1, s2).mean()
            p = logp.exp() * mask[:, i]
            ent = -(p * logp.clamp(min=-1e8)).sum(-1)
            entropy = entropy + ent.mean()
            v = net.critic(obs[:, i]).squeeze(-1)
            critic_loss = critic_loss + F.mse_loss(v, ret[:, i])
        actor_loss = actor_loss / n
        critic_loss = critic_loss / n
        entropy = entropy / n
        loss = actor_loss + self.ent_coef * entropy + self.vf_coef * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for net in self.nets for p in net.parameters()],
            self.max_grad_norm)
        self.opt.step()
        return dict(actor=float(actor_loss), critic=float(critic_loss),
                    entropy=float(entropy))


def train_ppo_local(cfg: SimConfig, episodes=20, steps=None, seed=0,
                    log_every=5, verbose=True):
    """Train the masked-PPO-with-local-critic baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    steps = steps or cfg.episode_len
    env = UAVIoTEnv(cfg)
    agent = PPOLocal(env.obs_dim, cfg.N)
    hist = []
    for ep in range(episodes):
        obs, state = env.reset()
        B = dict(obs=np.zeros((steps, cfg.N, env.obs_dim), dtype=np.float32),
                 mask=np.zeros((steps, cfg.N, N_ACTIONS), dtype=bool),
                 acts=np.zeros((steps, cfg.N), dtype=np.int64),
                 logp=np.zeros((steps, cfg.N), dtype=np.float32),
                 rew=np.zeros((steps, cfg.N), dtype=np.float32),
                 values=np.zeros((steps, cfg.N), dtype=np.float32))
        t = 0
        for _ in range(steps):
            mask = np.stack(env.action_masks())
            for i in range(cfg.N):
                if mask[i].sum() == 0:
                    mask[i, 0] = True
            acts, logp = agent.act(obs, mask)
            with torch.no_grad():
                vals = np.array([
                    float(agent.nets[i].critic(torch.as_tensor(
                        obs[i], dtype=torch.float32).unsqueeze(0))[0])
                    for i in range(cfg.N)])
            B["obs"][t] = obs
            B["mask"][t] = mask
            B["acts"][t] = acts
            B["logp"][t] = logp
            B["values"][t] = vals
            obs2, st, r, done, _ = env.step(acts)
            B["rew"][t] = r
            t += 1
            obs = obs2
            if done:
                break
        # GAE on the team reward, with the local critic
        adv = np.zeros_like(B["rew"])
        last = np.zeros(cfg.N, dtype=np.float32)
        for step in reversed(range(t)):
            next_v = B["values"][step + 1] if step + 1 < t else last
            delta = B["rew"][step] + agent.gamma * next_v - B["values"][step]
            last = delta + agent.gamma * agent.lam * last
            adv[step] = last
        # The advantage array is allocated for the full horizon and only the
        # first t rows are filled, so it must be truncated BEFORE the returns
        # are formed.  An episode that ends early -- which happens whenever a
        # UAV failure terminates the mission -- makes t < n_steps, and the
        # untruncated addition is a shape mismatch.
        ret = adv[:t] + B["values"][:t]
        B = {k: v[:t] for k, v in B.items()}
        info = agent.update(B, adv[:t], ret, B["logp"])
        s = env.summary()
        hist.append(dict(episode=ep, reward=float(B["rew"].sum()), **s))
        if verbose and (ep % max(1, log_every) == 0 or ep == episodes - 1):
            print(f"[PPO-Local ep {ep:3d}] R={hist[-1]['reward']:8.2f} "
                  f"CK={s.get('CK', 0):.3f} AG={s.get('AG', 0):.3f} "
                  f"ent={info.get('entropy', 0):.3f}", flush=True)
    return agent, hist


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def eval_id_dqn(agent, cfg, steps=None, seed=0):
    steps = steps or cfg.episode_len
    c = copy.deepcopy(cfg)
    c.seed = seed
    from .seeding import seed_evaluation
    seed_evaluation(seed)
    env = UAVIoTEnv(c)
    obs, _ = env.reset()
    for _ in range(steps):
        mask = np.stack(env.action_masks())
        for i in range(c.N):
            if mask[i].sum() == 0:
                mask[i, 0] = True
        acts = agent.act(obs, mask, deterministic=True)
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()


def eval_ppo_local(agent, cfg, steps=None, seed=0, deterministic=False):
    steps = steps or cfg.episode_len
    c = copy.deepcopy(cfg)
    c.seed = seed
    # this policy samples, so the global torch RNG must be seeded: see
    # uav_wireless/seeding.py
    from .seeding import seed_evaluation
    seed_evaluation(seed)
    env = UAVIoTEnv(c)
    obs, _ = env.reset()
    for _ in range(steps):
        mask = np.stack(env.action_masks())
        for i in range(c.N):
            if mask[i].sum() == 0:
                mask[i, 0] = True
        acts, _ = agent.act(obs, mask, deterministic=deterministic)
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()
