"""Agentes QMIX y red de mezcla para coordinación EW en formación (CTDE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F


class AgentRNN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)
        self.fc2 = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden)
        q: torch.Tensor = self.fc2(h)
        return q, h

    def init_hidden(self, batch: int) -> torch.Tensor:
        return self.fc1.weight.new_zeros(batch, self.hidden)


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int, hypernet_hidden: int) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, n_agents * embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(batch, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_tot: torch.Tensor = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch, 1)


@dataclass(frozen=True)
class QMIXConfig:
    hidden: int = 64
    mixer_embed_dim: int = 32
    hypernet_hidden: int = 64
    gamma: float = 0.99
    lr: float = 5e-4
    batch_episodes: int = 8
    buffer_episodes: int = 2000
    target_sync: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    learning_starts_episodes: int = 32
    double_q: bool = True
    grad_clip: float = 10.0


class QMIXLearner:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        state_dim: int,
        config: QMIXConfig,
        device: str,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = torch.device(device)
        self.rng = rng
        self.agent = AgentRNN(obs_dim, action_dim, config.hidden).to(self.device)
        self.target_agent = AgentRNN(obs_dim, action_dim, config.hidden).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, config.mixer_embed_dim, config.hypernet_hidden).to(
            self.device
        )
        self.target_mixer = QMixer(
            n_agents, state_dim, config.mixer_embed_dim, config.hypernet_hidden
        ).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.lr)
        self._updates = 0

    def init_hidden(self) -> dict[int, torch.Tensor]:
        return {a: self.agent.init_hidden(1) for a in range(self.n_agents)}

    def _mix(self, mixer: QMixer, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = agent_qs.shape
        flat_q = agent_qs.reshape(batch * horizon, self.n_agents)
        flat_s = states.reshape(batch * horizon, -1)
        mixed: torch.Tensor = mixer(flat_q, flat_s)
        return mixed.view(batch, horizon)

    def update(
        self,
        batch: tuple[
            NDArray[np.float32],
            NDArray[np.int64],
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.float32],
        ],
    ) -> float:
        obs_np, actions_np, rewards_np, states_np, dones_np, filled_np = batch
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device)
        filled = torch.as_tensor(filled_np, dtype=torch.float32, device=self.device)
        batch_size, horizon, n_agents, _ = obs.shape

        flat_obs = obs.permute(0, 2, 1, 3).reshape(batch_size * n_agents, horizon, -1)
        online_h = self.agent.init_hidden(batch_size * n_agents)
        target_h = self.target_agent.init_hidden(batch_size * n_agents)
        online_qs_t: list[torch.Tensor] = []
        target_qs_t: list[torch.Tensor] = []
        for t in range(horizon):
            oq, online_h = self.agent(flat_obs[:, t], online_h)
            tq, target_h = self.target_agent(flat_obs[:, t], target_h)
            online_qs_t.append(oq.view(batch_size, n_agents, -1))
            target_qs_t.append(tq.view(batch_size, n_agents, -1))
        online_qs = torch.stack(online_qs_t, dim=1)
        target_qs = torch.stack(target_qs_t, dim=1)

        chosen = torch.gather(online_qs, 3, actions.unsqueeze(3)).squeeze(3)
        if self.config.double_q:
            next_actions = online_qs.detach().argmax(dim=3, keepdim=True)
            target_max = torch.gather(target_qs, 3, next_actions).squeeze(3)
        else:
            target_max = target_qs.max(dim=3)[0]

        q_tot = self._mix(self.mixer, chosen, states)
        with torch.no_grad():
            target_tot = self._mix(self.target_mixer, target_max, states)
            y = rewards[:, :-1] + self.config.gamma * (1.0 - dones[:, :-1]) * target_tot[:, 1:]

        td_error = F.smooth_l1_loss(q_tot[:, :-1], y, reduction="none")
        mask = filled[:, :-1]
        loss = (td_error * mask).sum() / mask.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.config.target_sync == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(loss.item())

    @torch.no_grad()
    def select_actions(
        self,
        obs_dict: dict[int, NDArray[np.float32]],
        hidden: dict[int, torch.Tensor],
        epsilon: float,
    ) -> tuple[dict[int, int], dict[int, torch.Tensor]]:
        actions: dict[int, int] = {}
        new_hidden: dict[int, torch.Tensor] = {}
        for a in range(self.n_agents):
            obs_t = torch.as_tensor(obs_dict[a], dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            q, h = self.agent(obs_t, hidden[a])
            new_hidden[a] = h
            if self.rng.random() < epsilon:
                actions[a] = int(self.rng.integers(self.action_dim))
            else:
                actions[a] = int(torch.argmax(q, dim=1).item())
        return actions, new_hidden
