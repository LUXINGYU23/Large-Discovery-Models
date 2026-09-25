"""Task-local SAC training stack for improving-replay-buffers.

The pinned upstream task ships only an environment stub (online.py prints
"Implement or plug in your baseline training loop here"). This SAC loop,
evaluation, and result JSON are therefore part of the fixed task-local
evaluator, not the search space. A candidate replaces only the replay buffer.

    python rl_runner.py --domain cheetah --task run --seed 0 --steps 100000 \
        --buffer ldm_buffer --out results/ldm

--buffer names a module that defines class ReplayBuffer.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")


# ----------------------------------------------------------------- environment
class DMCEnv:
    """Flatten dm_control TimeSteps into (obs, reward, done).

    DMC episodes have a fixed 1000-step time limit and no terminal states, so
    truncation must never be treated as termination by the SAC bootstrap.
    """

    def __init__(self, domain: str, task: str, seed: int):
        from dm_control import suite

        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        spec = self._env.action_spec()
        self.action_dim = int(spec.shape[0])
        self.action_low = float(spec.minimum[0])
        self.action_high = float(spec.maximum[0])
        self.obs_dim = int(self._flatten(self._env.reset().observation).shape[0])
        self.max_steps = 1000

    @staticmethod
    def _flatten(observation) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
             for v in observation.values()]
        ).astype(np.float32)

    def reset(self) -> np.ndarray:
        return self._flatten(self._env.reset().observation)

    def step(self, action: np.ndarray):
        ts = self._env.step(np.clip(action, self.action_low, self.action_high))
        reward = float(ts.reward or 0.0)
        return self._flatten(ts.observation), reward, bool(ts.last())


# ----------------------------------------------------------------- SAC
def mlp(sizes: list[int], out_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0


class Actor(nn.Module):
    """Squashed Gaussian policy with the tanh log-probability Jacobian correction."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], out_act=True)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            return torch.tanh(mu), None
        dist = torch.distributions.Normal(mu, std)
        pre = dist.rsample()
        action = torch.tanh(pre)
        logp = dist.log_prob(pre).sum(-1)
        logp -= (2 * (np.log(2.0) - pre - F.softplus(-2 * pre))).sum(-1)
        return action, logp


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005):
        self.device = device
        self.gamma, self.tau = gamma, tau
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # Automatic entropy temperature with the conventional -|A| target.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch) -> dict[str, float]:
        obs, act, rew, next_obs, done = batch
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_act)
            target_v = torch.min(tq1, tq2) - self.alpha.detach() * next_logp
            target_q = rew + self.gamma * (1.0 - done) * target_v

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_act, logp = self.actor(obs)
        aq1, aq2 = self.critic(obs, new_act)
        actor_loss = (self.alpha.detach() * logp - torch.min(aq1, aq2)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss),
                "alpha": float(self.alpha)}


# ----------------------------------------------------------------- evaluation and loop
def evaluate(env: DMCEnv, agent: SAC, episodes: int = 10) -> float:
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            obs, reward, done = env.step(agent.act(obs, deterministic=True))
            total += reward
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="cheetah")
    parser.add_argument("--task", default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--start-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--buffer", default="ldm_buffer",
                        help="module defining class ReplayBuffer")
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DMCEnv(args.domain, args.task, args.seed)
    eval_env = DMCEnv(args.domain, args.task, args.seed + 10_000)
    agent = SAC(env.obs_dim, env.action_dim, device)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    buffer_cls = getattr(importlib.import_module(args.buffer), "ReplayBuffer")
    buffer = buffer_cls(obs_dim=env.obs_dim, act_dim=env.action_dim,
                        capacity=args.capacity, device=device)

    obs = env.reset()
    episode_return, episode_len = 0.0, 0
    curve: list[dict[str, float]] = []
    started = time.time()

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            action = np.random.uniform(env.action_low, env.action_high, env.action_dim)
        else:
            action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        episode_return += reward
        episode_len += 1
        # DMC has time limits but no true terminals; storing done=1 would truncate values.
        buffer.add(obs, action, reward, next_obs, 0.0)
        obs = next_obs
        if done or episode_len >= env.max_steps:
            obs = env.reset()
            episode_return, episode_len = 0.0, 0

        if step > args.start_steps and len(buffer) >= args.batch_size:
            agent.update(buffer.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.steps:
            score = evaluate(eval_env, agent, args.eval_episodes)
            curve.append({"step": step, "average_return": score})
            print("step %6d  return %7.2f  elapsed %5.1f min  buffer %d"
                  % (step, score, (time.time() - started) / 60, len(buffer)), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": f"{args.domain}-{args.task}-v0",
        "seed": args.seed,
        "buffer": args.buffer,
        "steps": args.steps,
        "metrics": {"average_return": curve[-1]["average_return"] if curve else 0.0},
        "curve": curve,
        "wall_clock_minutes": round((time.time() - started) / 60, 2),
    }
    path = out_dir / f"{args.domain}-{args.task}_seed{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
