"""Seed replay buffer: the uniform circular buffer the task asks to improve.

Interface: ReplayBuffer(obs_dim, act_dim, capacity, device); add(...) -> None once
per environment step; sample(batch_size) -> (obs, act, rew, next_obs, done)
float32 tensors on device with shapes (B, obs_dim) (B, act_dim) (B,) (B, obs_dim) (B,);
__len__() -> number of stored transitions.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.idx
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)
