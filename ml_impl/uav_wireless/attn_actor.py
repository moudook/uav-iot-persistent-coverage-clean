"""Graph-attention actor (Innovation 6, TAG-MAPPO style).

Replaces the fixed 4-neighbor-slot MLP actor with a small multi-head
self-attention block over feature tokens.  The tokenisation is fixed and
configuration-independent: the observation vector is split into a small
number of named tokens (own state, neighbor slots, coverage map, tail) and
each token is projected to a common width before attention.  This makes the
policy permutation-invariant over the neighbor slots and lets it aggregate
information across the whole local observation instead of reading a fixed
ordering.

Usage:  MAPPO(..., attn=True)  or pass ActorAttention as the actor class.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .mappo import N_ACTIONS


class Tokeniser(nn.Module):
    """Split a flat obs vector into named tokens and project to width d."""

    def __init__(self, obs_dim, d=64, n_tokens=8):
        super().__init__()
        self.d = d
        self.n_tokens = n_tokens
        # learnable projection of the whole obs into n_tokens x d
        self.proj = nn.Linear(obs_dim, n_tokens * d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))

    def forward(self, obs):
        # obs: (B, obs_dim) -> (B, n_tokens, d)
        x = self.proj(obs).view(obs.shape[0], self.n_tokens, self.d)
        return x + self.pos


class ActorAttention(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_heads=4, n_tokens=8):
        super().__init__()
        d = hidden // 2  # token width
        self.tok = Tokeniser(obs_dim, d=d, n_tokens=n_tokens)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads,
                                          batch_first=True, dropout=0.0)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(),
                                nn.Linear(d * 2, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d * n_tokens, N_ACTIONS)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)

    def forward(self, obs):
        x = self.tok(obs)                       # (B, T, d)
        a, _ = self.attn(x, x, x)              # self-attention
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        x = x.flatten(1)                       # (B, T*d)
        return self.head(x)


def make_actor(obs_dim, hidden=128, attn=False):
    if attn:
        return ActorAttention(obs_dim, hidden=hidden)
    from .mappo import Actor
    return Actor(obs_dim, hidden)
