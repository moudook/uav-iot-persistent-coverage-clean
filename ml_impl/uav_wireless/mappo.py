"""
CTDE MAPPO for UAV-IoT persistent coverage. Section IV-A / Algorithm 1.

- Shared actor  pi_theta(a_i | o_i)   with masked logits  (deployment-ready)
- Centralized critic V_phi(s_t)      (training only)
- GAE (26), clipped PPO (28), critic loss (29)
- Parameter sharing across UAVs: one actor, one critic, N agents.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import N_ACTIONS, UAVIoTEnv, SimConfig


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    if out_act is not None:
        layers[-1] = nn.Linear(sizes[-2], sizes[-1])
        layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Shared policy with masked softmax over the 6 discrete actions."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], nn.ReLU, None)
        self.head = nn.Linear(hidden, N_ACTIONS)
        # orthogonal init (MAPPO/CleanRL default) for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)

    def forward(self, obs):
        return self.head(self.net(obs))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = mlp([state_dim, hidden, hidden], nn.ReLU, None)
        self.head = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)

    def forward(self, s):
        return self.head(self.net(s)).squeeze(-1)


class MAPPO:
    def __init__(
        self,
        obs_dim: int | None = None,
        state_dim: int | None = None,
        n_agents=5,
        hidden=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        per_agent_adv_norm=False,
        value_norm=True,
        device="cpu",
        attn=False,
    ):
        # Dimensions are configuration-dependent (they scale with N, M and
        # n_pads). Falling back to the reference-config constants keeps old
        # call sites working, but callers should pass env.obs_dim and
        # env.state_dim so the networks always match the environment.
        obs_dim = UAVIoTEnv.OBS_DIM if obs_dim is None else obs_dim
        state_dim = UAVIoTEnv.STATE_DIM if state_dim is None else state_dim
        self.device = torch.device(device)
        from .attn_actor import make_actor
        self.actor = make_actor(obs_dim, hidden, attn=attn).to(self.device)
        self.critic = Critic(state_dim, hidden).to(self.device)
        # Separate optimisers + separate grad clipping (SI feedback 1.3).
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_agents = n_agents
        # Per-agent advantage normalisation.  With a shared team reward every
        # agent receives the SAME advantage, so the policy gradient cannot
        # distinguish an agent whose action helped from one whose action was
        # irrelevant.  Normalising each agent's advantages over its own
        # history gives every agent a signal relative to its own behaviour
        # rather than to the pooled batch.  (Generalised per-agent advantage
        # estimation, Kim et al., AAMAS 2026, addresses the same problem for
        # the multi-agent case.)
        self.per_agent_adv_norm = per_agent_adv_norm
        self.attn = attn
        # MAPPO's value normalisation.  The critic predicts the returns in a
        # normalised space defined by running statistics of the return
        # distribution; the value is un-normalised at every point of use.
        self.value_norm = value_norm
        self.ret_ema_beta = 0.99
        # KL anchor attributes (optional; used by residual/BC experiments)
        self.ref_actor = None
        self.kl_coef = 0.0
        self.kl_target = 0.01
        self.ent_floor = 0.0
        self.ent_floor_coef = 0.0
        self._beta = 0.0
        # floor on the standard deviation of the returns, so that a policy
        # with nearly constant returns cannot drive the normalisation to
        # zero and collapse the advantage
        self.ret_std_floor = 1.0
        self._ret_mean = 0.0
        self._ret_std = 1.0

    # ------------------------------------------------------------------
    def masked_logits(self, obs, mask):
        logits = self.actor(obs)
        return logits.masked_fill(~mask, float("-inf"))

    @torch.no_grad()
    def act(self, obs_np, mask_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
        logits = self.masked_logits(obs, mask)
        # a fully-masked row would give NaNs: fall back to hover (index 0)
        bad = torch.isinf(logits).all(dim=-1)
        if bad.any():
            logits[bad] = torch.zeros_like(logits[bad])
            logits[bad, 0] = 1.0
        logp = torch.log_softmax(logits, dim=-1)
        if deterministic:
            a = logp.argmax(dim=-1)
        else:
            a = torch.distributions.Categorical(logits=logp).sample()
        return a.cpu().numpy()

    # ------------------------------------------------------------------
    def value(self, states):
        """The value of a batch of global states, in the ORIGINAL reward scale.

        With value normalisation the critic's raw output is the NORMALISED
        value, so every place that uses V(s) -- the TD residual, the GAE
        recursion and the bootstrap at a truncation boundary -- must
        un-normalise it first.  Forgetting this is the classic way to
        introduce a silent scale error into the advantage estimates.
        """
        v = self.critic(states)
        if self.value_norm:
            v = v * self._ret_std + self._ret_mean
        return v


    def update(self, batch, epochs=10, n_minibatch=4, target_kl=0.02):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.bool, device=self.device)
        acts = torch.as_tensor(batch["acts"], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        old_val = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)
        if "active" in batch:
            active = torch.as_tensor(batch["active"], dtype=torch.bool, device=self.device)
        else:
            active = torch.ones_like(mask[:, :, 0], dtype=torch.bool)

        n_agents = acts.shape[1]
        T = acts.shape[0]
        ret_c = ret.reshape(-1)
        old_val_c = old_val.reshape(-1)

        if self.per_agent_adv_norm:
            adv = (adv - adv.mean(dim=0, keepdim=True)) / (
                adv.std(dim=0, keepdim=True) + 1e-8)
        else:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # running value-norm stats updated once per rollout
        if self.value_norm:
            with torch.no_grad():
                b = self.ret_ema_beta
                # b weights the OLD estimate (smoothing), (1-b) the new batch
                self._ret_mean = b * self._ret_mean + (1 - b) * float(ret_c.mean())
                sd = float(ret_c.std())
                self._ret_std = max(b * self._ret_std + (1 - b) * sd,
                                    self.ret_std_floor)

        if self.value_norm:
            target_c = (ret_c - self._ret_mean) / (self._ret_std + 1e-8)
            old_val_c_n = (old_val_c - self._ret_mean) / (self._ret_std + 1e-8)
        else:
            target_c = ret_c
            old_val_c_n = old_val_c

        total = T * n_agents
        active_flat = active.reshape(-1)
        last = {}
        for epoch in range(epochs):
            perm = torch.randperm(total, device=self.device)
            kl_accum = 0.0
            n_mb = 0
            for mb in torch.chunk(perm, n_minibatch):
                if mb.numel() == 0:
                    continue
                t_idx = mb // n_agents
                i_idx = mb % n_agents

                mb_obs = obs[t_idx, i_idx]
                mb_mask = mask[t_idx, i_idx]
                mb_acts = acts[t_idx, i_idx]
                mb_old_logp = old_logp[t_idx, i_idx]
                mb_adv = adv[t_idx, i_idx]
                mb_target = target_c[mb]
                mb_old_val = old_val_c_n[mb]

                logits = self.masked_logits(mb_obs, mb_mask)
                logp = torch.log_softmax(logits, dim=-1)
                logp_a = logp.gather(-1, mb_acts.unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(logp_a - mb_old_logp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                per_sample_actor = -torch.min(s1, s2)
                w = active_flat[mb].float()
                if w.sum() < 1:
                    w = torch.ones_like(w)
                actor_loss = (per_sample_actor * w).sum() / w.sum()

                # entropy (positive H)
                p_ = logp.exp() * mb_mask
                ent = -(p_ * logp.clamp(min=-1e8)).sum(-1)
                ent = torch.where(mb_mask.any(-1), ent, torch.zeros_like(ent))
                w_ent = active_flat[mb].float()
                if w_ent.sum() < 1:
                    w_ent = torch.ones_like(w_ent)
                ent = (ent * w_ent).sum() / w_ent.sum()

                # critic (clipped value loss)
                mb_states = states[t_idx]
                new_val = self.critic(mb_states)
                w_c = active_flat[mb].float()
                if w_c.sum() < 1:
                    w_c = torch.ones_like(w_c)
                if self.value_norm:
                    v_clipped = mb_old_val + torch.clamp(new_val - mb_old_val, -self.clip, self.clip)
                    v_loss1 = (new_val - mb_target) ** 2
                    v_loss2 = (v_clipped - mb_target) ** 2
                    per_c = torch.max(v_loss1, v_loss2)
                    critic_loss = (per_c * w_c).sum() / w_c.sum()
                else:
                    per_c = (new_val - mb_target) ** 2
                    critic_loss = (per_c * w_c).sum() / w_c.sum()

                # KL anchor to reference actor
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.ref_actor is not None:
                    with torch.no_grad():
                        ref_logits = self.ref_actor(mb_obs)
                        ref_logits = ref_logits.masked_fill(~mb_mask, float("-inf"))
                        ref_logp = torch.log_softmax(ref_logits, dim=-1)
                    p_ref = ref_logp.exp() * mb_mask
                    kl = (p_ref * (ref_logp.clamp(min=-1e8) - logp.clamp(min=-1e8))).sum(-1)
                    kl = torch.where(mb_mask.any(-1), kl, torch.zeros_like(kl)).mean()
                    kl_loss = self._beta * kl
                    if kl.item() > 1.5 * self.kl_target:
                        self._beta = min(self._beta * 2.0, 50.0)
                    elif kl.item() < 0.5 * self.kl_target:
                        self._beta = max(self._beta * 0.5, 0.05)

                ent_floor_loss = torch.tensor(0.0, device=self.device)
                if self.ent_floor_coef > 0 and self.ent_floor > 0:
                    ent_floor_loss = self.ent_floor_coef * torch.clamp(
                        self.ent_floor - ent, min=0.0
                    )

                # entropy sign FIXED: subtract entropy bonus (maximise entropy)
                loss = (actor_loss - self.ent_coef * ent
                        + self.vf_coef * critic_loss
                        + kl_loss + ent_floor_loss)

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                with torch.no_grad():
                    log_ratio = logp_a - mb_old_logp
                    approx_kl = (torch.exp(log_ratio) - 1 - log_ratio).mean().item()
                kl_accum += abs(approx_kl)
                n_mb += 1
                last = dict(actor=float(actor_loss), critic=float(critic_loss),
                            entropy=float(ent), value=float(new_val.mean()),
                            kl=float(kl_loss) if self.ref_actor is not None else 0.0,
                            beta=float(self._beta) if self.ref_actor is not None else 0.0)

            if n_mb > 0 and (kl_accum / n_mb) > target_kl:
                break

        if not last:
            last = dict(actor=0.0, critic=0.0, entropy=0.0, value=0.0, kl=0.0, beta=0.0)
        return last
    def save(self, path):
        torch.save(
            dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                 value_norm=self.value_norm,
                 ret_mean=self._ret_mean, ret_std=self._ret_std),
            path,
        )

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        if "ret_mean" in d:
            self._ret_mean = float(d["ret_mean"])
            self._ret_std = float(d["ret_std"])