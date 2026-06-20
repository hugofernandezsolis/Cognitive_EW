# QMIX + entrenamiento CTDE (Modelo 3, sub-pieza B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar una formación de 4 agentes EW con QMIX (CTDE) sobre el entorno IADS ya existente, de forma reproducible y con latencia de ejecución descentralizada documentada.

**Architecture:** Red de agente recurrente (GRU/DRQN) con parámetros compartidos + red de mezcla monótona QMIX condicionada al estado global. Entrenamiento centralizado (mixer ve el estado global), ejecución descentralizada (cada agente corre su GRU). Replay por episodios + BPTT, Double-DQN con target mixer, Huber loss enmascarada por longitud de episodio.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML. Tests con pytest. Reutiliza `IADSFormationEnv`/`IADSEnvConfig` (`src/cog_ew/marl_formation/env.py`) y `profile_latency` (`src/cog_ew/temporal_cnn_elint/metrics.py`).

## Global Constraints

- **Type hints** en todas las funciones públicas; mypy `--strict` debe pasar (`loss.backward()  # type: ignore[no-untyped-call]`).
- **ruff check** y **ruff format** limpios antes de cada commit (line-length 100).
- **Reproducibilidad:** seeds explícitos (`random`, `numpy`, `torch`); hiperparámetros SOLO en YAML versionado; `run_meta.json` con seed + hyperparams + hash de config + versiones.
- **No hardcodear hiperparámetros** en el código.
- **Seguridad EW:** no exponer parámetros de amenazas reales en logs/artefactos; si se recarga un checkpoint, `torch.load(..., weights_only=True)`.
- **Herramientas:** ejecutar tests con `.venv/bin/python -m pytest`; ruff/mypy con `.venv/bin/ruff` y `.venv/bin/mypy`.
- **Commits** terminan con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Dimensiones del entorno (sub-pieza A):** `obs_dim = n_radars*5 + n_agents`, `state_dim = n_radars*6 + n_agents`, `action_dim = n_radars*3*len(power_levels)`. Con el `env.yaml` actual (M=4, N=4, P=4): `obs_dim=24`, `state_dim=28`, `action_dim=48`.

---

### Task 1: `AgentRNN` — red de agente recurrente compartida

**Files:**
- Create: `src/cog_ew/marl_formation/agents.py`
- Test: `tests/marl_formation/test_agents.py`

**Interfaces:**
- Consumes: nada (primera tarea del módulo).
- Produces: `AgentRNN(obs_dim: int, action_dim: int, hidden: int)` con `forward(obs: Tensor (B,obs_dim), hidden: Tensor (B,hidden)) -> tuple[Tensor (B,action_dim), Tensor (B,hidden)]` y `init_hidden(batch: int) -> Tensor (batch,hidden)`.

- [ ] **Step 1: Write the failing test**

```python
import torch

from cog_ew.marl_formation.agents import AgentRNN


def test_agent_rnn_forward_shapes():
    net = AgentRNN(obs_dim=24, action_dim=48, hidden=32)
    obs = torch.zeros(5, 24)
    h0 = net.init_hidden(5)
    q, h1 = net(obs, h0)
    assert q.shape == (5, 48)
    assert h1.shape == (5, 32)


def test_agent_rnn_is_deterministic_by_seed():
    torch.manual_seed(0)
    a = AgentRNN(24, 48, 32)
    torch.manual_seed(0)
    b = AgentRNN(24, 48, 32)
    obs = torch.ones(3, 24)
    qa, _ = a(obs, a.init_hidden(3))
    qb, _ = b(obs, b.init_hidden(3))
    assert torch.allclose(qa, qb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'AgentRNN'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Agentes QMIX y red de mezcla para coordinación EW en formación (CTDE)."""

from __future__ import annotations

import torch
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py && .venv/bin/ruff check src/cog_ew/marl_formation/agents.py && .venv/bin/mypy src/cog_ew/marl_formation/agents.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py
git commit -m "$(cat <<'EOF'
feat(marl): AgentRNN (red de agente GRU compartida para QMIX)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `QMixer` — red de mezcla monótona

**Files:**
- Modify: `src/cog_ew/marl_formation/agents.py`
- Test: `tests/marl_formation/test_agents.py`

**Interfaces:**
- Consumes: nada nuevo.
- Produces: `QMixer(n_agents: int, state_dim: int, embed_dim: int, hypernet_hidden: int)` con `forward(agent_qs: Tensor (B,n_agents), state: Tensor (B,state_dim)) -> Tensor (B,1)`. Garantiza `∂Q_tot/∂Q_i ≥ 0` (pesos por `abs()`).

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.marl_formation.agents import QMixer


def test_qmixer_output_shape():
    mixer = QMixer(n_agents=4, state_dim=28, embed_dim=16, hypernet_hidden=32)
    agent_qs = torch.zeros(7, 4)
    state = torch.zeros(7, 28)
    q_tot = mixer(agent_qs, state)
    assert q_tot.shape == (7, 1)


def test_qmixer_is_monotonic_in_agent_qs():
    torch.manual_seed(0)
    mixer = QMixer(4, 28, 16, 32)
    agent_qs = torch.randn(3, 4, requires_grad=True)
    state = torch.randn(3, 28)
    q_tot = mixer(agent_qs, state).sum()
    grad = torch.autograd.grad(q_tot, agent_qs)[0]
    assert torch.all(grad >= -1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k qmixer -v`
Expected: FAIL with `ImportError: cannot import name 'QMixer'`.

- [ ] **Step 3: Write minimal implementation**

Añade a `agents.py` (después de `AgentRNN`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k qmixer -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py && .venv/bin/ruff check src/cog_ew/marl_formation/agents.py && .venv/bin/mypy src/cog_ew/marl_formation/agents.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py
git commit -m "$(cat <<'EOF'
feat(marl): QMixer (red de mezcla monótona condicionada al estado global)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `QMIXConfig` + `QMIXLearner` (init, `init_hidden`, `select_actions`)

**Files:**
- Modify: `src/cog_ew/marl_formation/agents.py`
- Test: `tests/marl_formation/test_agents.py`

**Interfaces:**
- Consumes: `AgentRNN`, `QMixer`.
- Produces:
  - `QMIXConfig` (dataclass frozen) con campos: `hidden`, `mixer_embed_dim`, `hypernet_hidden`, `gamma`, `lr`, `batch_episodes`, `buffer_episodes`, `target_sync`, `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`, `learning_starts_episodes`, `double_q`, `grad_clip`.
  - `QMIXLearner(obs_dim, action_dim, n_agents, state_dim, config, device, rng)` con `init_hidden() -> dict[int, Tensor]` y `select_actions(obs_dict: dict[int, NDArray], hidden: dict[int, Tensor], epsilon: float) -> tuple[dict[int, int], dict[int, Tensor]]`. Expone `self.agent: AgentRNN`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from cog_ew.marl_formation.agents import QMIXConfig, QMIXLearner


def _learner() -> QMIXLearner:
    return QMIXLearner(
        obs_dim=24,
        action_dim=48,
        n_agents=4,
        state_dim=28,
        config=QMIXConfig(hidden=16, mixer_embed_dim=8, hypernet_hidden=16),
        device="cpu",
        rng=np.random.default_rng(0),
    )


def test_select_actions_returns_valid_dict():
    learner = _learner()
    obs = {a: np.zeros(24, dtype=np.float32) for a in range(4)}
    actions, hidden = learner.select_actions(obs, learner.init_hidden(), epsilon=0.0)
    assert set(actions) == {0, 1, 2, 3}
    assert all(0 <= actions[a] < 48 for a in range(4))
    assert set(hidden) == {0, 1, 2, 3}


def test_select_actions_greedy_is_deterministic():
    learner = _learner()
    obs = {a: np.ones(24, dtype=np.float32) for a in range(4)}
    a1, _ = learner.select_actions(obs, learner.init_hidden(), epsilon=0.0)
    a2, _ = learner.select_actions(obs, learner.init_hidden(), epsilon=0.0)
    assert a1 == a2


def test_select_actions_full_epsilon_explores():
    learner = _learner()
    obs = {a: np.zeros(24, dtype=np.float32) for a in range(4)}
    actions, _ = learner.select_actions(obs, learner.init_hidden(), epsilon=1.0)
    assert all(0 <= actions[a] < 48 for a in range(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k select_actions -v`
Expected: FAIL with `ImportError: cannot import name 'QMIXConfig'`.

- [ ] **Step 3: Write minimal implementation**

Añade los imports al principio de `agents.py`:

```python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
```

Añade tras `QMixer`:

```python
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
            obs_t = torch.as_tensor(
                obs_dict[a], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q, h = self.agent(obs_t, hidden[a])
            new_hidden[a] = h
            if self.rng.random() < epsilon:
                actions[a] = int(self.rng.integers(self.action_dim))
            else:
                actions[a] = int(torch.argmax(q, dim=1).item())
        return actions, new_hidden
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k select_actions -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py && .venv/bin/ruff check src/cog_ew/marl_formation/agents.py && .venv/bin/mypy src/cog_ew/marl_formation/agents.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py
git commit -m "$(cat <<'EOF'
feat(marl): QMIXConfig + QMIXLearner (init + selección ε-greedy descentralizada)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `QMIXLearner.update` — Double-DQN + target mixer + Huber enmascarada

**Files:**
- Modify: `src/cog_ew/marl_formation/agents.py`
- Test: `tests/marl_formation/test_agents.py`

**Interfaces:**
- Consumes: `QMIXLearner` (Task 3).
- Produces: `QMIXLearner.update(batch) -> float`, donde `batch` es una tupla de `NDArray`:
  `(obs (B,T,N,obs_dim), actions (B,T,N) int64, rewards (B,T), states (B,T,state_dim), dones (B,T), filled (B,T))`. Devuelve la loss escalar.

- [ ] **Step 1: Write the failing test**

```python
def _batch(B=2, T=5, N=4, obs_dim=24, state_dim=28, action_dim=48):
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((B, T, N, obs_dim)).astype(np.float32)
    actions = rng.integers(0, action_dim, size=(B, T, N)).astype(np.int64)
    rewards = rng.standard_normal((B, T)).astype(np.float32)
    states = rng.standard_normal((B, T, state_dim)).astype(np.float32)
    dones = np.zeros((B, T), dtype=np.float32)
    dones[:, -1] = 1.0
    filled = np.ones((B, T), dtype=np.float32)
    return obs, actions, rewards, states, dones, filled


def test_update_returns_finite_loss_and_changes_params():
    learner = _learner()
    before = next(learner.agent.parameters()).clone()
    loss = learner.update(_batch())
    after = next(learner.agent.parameters())
    assert np.isfinite(loss)
    assert not torch.allclose(before, after)


def test_update_syncs_target_after_interval():
    config = QMIXConfig(hidden=16, mixer_embed_dim=8, hypernet_hidden=16, target_sync=1)
    learner = QMIXLearner(24, 48, 4, 28, config, "cpu", np.random.default_rng(0))
    learner.update(_batch())
    for online, target in zip(
        learner.agent.parameters(), learner.target_agent.parameters(), strict=True
    ):
        assert torch.allclose(online, target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k update -v`
Expected: FAIL with `AttributeError: 'QMIXLearner' object has no attribute 'update'`.

- [ ] **Step 3: Write minimal implementation**

Añade el método `_mix` y `update` a `QMIXLearner`:

```python
    def _mix(
        self, mixer: QMixer, agent_qs: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        batch, horizon, _ = agent_qs.shape
        flat_q = agent_qs.reshape(batch * horizon, self.n_agents)
        flat_s = states.reshape(batch * horizon, -1)
        return mixer(flat_q, flat_s).view(batch, horizon)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -v`
Expected: PASS (todos los tests de agents).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py && .venv/bin/ruff check src/cog_ew/marl_formation/agents.py && .venv/bin/mypy src/cog_ew/marl_formation/agents.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py
git commit -m "$(cat <<'EOF'
feat(marl): QMIXLearner.update (Double-DQN + target mixer + Huber enmascarada)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `EpisodeReplayBuffer` — replay por episodios padeados

**Files:**
- Create: `src/cog_ew/marl_formation/train.py`
- Test: `tests/marl_formation/test_train.py`

**Interfaces:**
- Consumes: nada nuevo.
- Produces: `EpisodeReplayBuffer(capacity, horizon, n_agents, obs_dim, state_dim)` con:
  - `add(obs (T,N,obs_dim), actions (T,N), rewards (T,), states (T,state_dim), dones (T,), filled (T,)) -> None`
  - `sample(batch_episodes, rng) -> tuple[NDArray, ...]` con shapes `(B,T,...)` en el mismo orden que `update`.
  - `__len__() -> int`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from cog_ew.marl_formation.train import EpisodeReplayBuffer


def test_episode_buffer_add_and_sample_shapes():
    buf = EpisodeReplayBuffer(capacity=10, horizon=6, n_agents=4, obs_dim=24, state_dim=28)
    for _ in range(3):
        buf.add(
            obs=np.zeros((6, 4, 24), dtype=np.float32),
            actions=np.zeros((6, 4), dtype=np.int64),
            rewards=np.zeros(6, dtype=np.float32),
            states=np.zeros((6, 28), dtype=np.float32),
            dones=np.zeros(6, dtype=np.float32),
            filled=np.ones(6, dtype=np.float32),
        )
    assert len(buf) == 3
    obs, actions, rewards, states, dones, filled = buf.sample(2, np.random.default_rng(0))
    assert obs.shape == (2, 6, 4, 24)
    assert actions.shape == (2, 6, 4)
    assert rewards.shape == (2, 6)
    assert states.shape == (2, 6, 28)
    assert filled.shape == (2, 6)


def test_episode_buffer_respects_capacity():
    buf = EpisodeReplayBuffer(capacity=2, horizon=4, n_agents=4, obs_dim=24, state_dim=28)
    for _ in range(5):
        buf.add(
            obs=np.zeros((4, 4, 24), dtype=np.float32),
            actions=np.zeros((4, 4), dtype=np.int64),
            rewards=np.zeros(4, dtype=np.float32),
            states=np.zeros((4, 28), dtype=np.float32),
            dones=np.zeros(4, dtype=np.float32),
            filled=np.ones(4, dtype=np.float32),
        )
    assert len(buf) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'EpisodeReplayBuffer'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Entrenamiento CTDE (QMIX) de la formación EW sobre el entorno IADS."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class EpisodeReplayBuffer:
    def __init__(
        self, capacity: int, horizon: int, n_agents: int, obs_dim: int, state_dim: int
    ) -> None:
        self.capacity = capacity
        self._obs = np.zeros((capacity, horizon, n_agents, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, horizon, n_agents), dtype=np.int64)
        self._rewards = np.zeros((capacity, horizon), dtype=np.float32)
        self._states = np.zeros((capacity, horizon, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, horizon), dtype=np.float32)
        self._filled = np.zeros((capacity, horizon), dtype=np.float32)
        self._size = 0
        self._pos = 0

    def add(
        self,
        obs: NDArray[np.float32],
        actions: NDArray[np.int64],
        rewards: NDArray[np.float32],
        states: NDArray[np.float32],
        dones: NDArray[np.float32],
        filled: NDArray[np.float32],
    ) -> None:
        i = self._pos
        self._obs[i] = obs
        self._actions[i] = actions
        self._rewards[i] = rewards
        self._states[i] = states
        self._dones[i] = dones
        self._filled[i] = filled
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_episodes: int, rng: np.random.Generator
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        idx = rng.integers(0, self._size, size=batch_episodes)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._states[idx],
            self._dones[idx],
            self._filled[idx],
        )

    def __len__(self) -> int:
        return self._size
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_train.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/train.py tests/marl_formation/test_train.py && .venv/bin/ruff check src/cog_ew/marl_formation/train.py && .venv/bin/mypy src/cog_ew/marl_formation/train.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/train.py tests/marl_formation/test_train.py
git commit -m "$(cat <<'EOF'
feat(marl): EpisodeReplayBuffer (replay por episodios padeados para QMIX)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `TrainConfig` + `qmix.yaml` + bucle `train` (CTDE end-to-end)

**Files:**
- Modify: `src/cog_ew/marl_formation/train.py`
- Create: `configs/marl_formation/qmix.yaml`
- Test: `tests/marl_formation/test_train.py`

**Interfaces:**
- Consumes: `EpisodeReplayBuffer` (Task 5), `QMIXConfig`/`QMIXLearner` (Tasks 3-4), `IADSEnvConfig`/`IADSFormationEnv` (`env.py`), `profile_latency` (`temporal_cnn_elint.metrics`).
- Produces:
  - `TrainConfig` (dataclass + `from_yaml(path) -> TrainConfig`) con `env: IADSEnvConfig`, `agent: QMIXConfig`, `total_episodes`, `eval_episodes`, `eval_every`, `device`, `seed`, `out_dir`, `tracking`.
  - `train(config: TrainConfig) -> dict[str, Any]` que escribe `run_meta.json`, `metrics.json`, `best.pt` y devuelve `{"win_rate_history": [...], "final": {...}}`.

- [ ] **Step 1: Write the failing test**

```python
from cog_ew.marl_formation.train import TrainConfig, train

CONFIG = "configs/marl_formation/qmix.yaml"


def test_train_config_from_yaml_parses_sections():
    config = TrainConfig.from_yaml(CONFIG)
    assert config.env.n_agents == 4
    assert config.agent.gamma == 0.99
    assert config.total_episodes > 0


def test_train_smoke_produces_metrics(tmp_path):
    config = TrainConfig.from_yaml(CONFIG)
    config = replace_for_smoke(config, tmp_path)
    result = train(config)
    assert 0.0 <= result["final"]["win_rate"] <= 1.0
    assert np.isfinite(result["final"]["latency_mean_ms"])
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def replace_for_smoke(config: TrainConfig, tmp_path) -> TrainConfig:
    from dataclasses import replace

    from cog_ew.marl_formation.agents import QMIXConfig

    agent = replace(config.agent, learning_starts_episodes=2, batch_episodes=2)
    return replace(
        config,
        agent=agent,
        total_episodes=6,
        eval_episodes=3,
        eval_every=3,
        out_dir=str(tmp_path),
    )


def test_train_is_deterministic_by_seed(tmp_path):
    a = train(replace_for_smoke(TrainConfig.from_yaml(CONFIG), tmp_path / "a"))
    b = train(replace_for_smoke(TrainConfig.from_yaml(CONFIG), tmp_path / "b"))
    assert a["win_rate_history"] == b["win_rate_history"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_train.py -k "train" -v`
Expected: FAIL with `ImportError: cannot import name 'TrainConfig'` (y falta `configs/marl_formation/qmix.yaml`).

- [ ] **Step 3a: Create the config**

`configs/marl_formation/qmix.yaml`:

```yaml
env_config: configs/marl_formation/env.yaml
agent:
  hidden: 64
  mixer_embed_dim: 32
  hypernet_hidden: 64
  gamma: 0.99
  lr: 0.0005
  batch_episodes: 8
  buffer_episodes: 2000
  target_sync: 200
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 2000
  learning_starts_episodes: 32
  double_q: true
  grad_clip: 10.0
total_episodes: 4000
eval_episodes: 50
eval_every: 200
device: cpu
seed: 0
out_dir: runs/marl_formation
tracking: false
```

- [ ] **Step 3b: Write minimal implementation**

Añade los imports al principio de `train.py` (junto a los existentes):

```python
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium
import torch
import yaml

from cog_ew.marl_formation.agents import AgentRNN, QMIXConfig, QMIXLearner
from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv
from cog_ew.temporal_cnn_elint.metrics import profile_latency
```

Añade tras `EpisodeReplayBuffer`:

```python
@dataclass
class TrainConfig:
    env: IADSEnvConfig
    agent: QMIXConfig
    total_episodes: int = 4000
    eval_episodes: int = 50
    eval_every: int = 200
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/marl_formation"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        env = IADSEnvConfig.from_yaml(raw.pop("env_config"))
        agent = QMIXConfig(**raw.pop("agent"))
        return cls(env=env, agent=agent, **raw)


class _AgentForward(torch.nn.Module):
    def __init__(self, agent: AgentRNN) -> None:
        super().__init__()
        self.agent = agent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.agent.init_hidden(obs.size(0))
        q, _ = self.agent(obs, h)
        return q


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epsilon(episode: int, config: QMIXConfig) -> float:
    frac = min(1.0, episode / config.epsilon_decay_steps)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def _run_metadata(config: TrainConfig) -> dict[str, Any]:
    hyperparameters = asdict(config)
    blob = json.dumps(hyperparameters, sort_keys=True).encode()
    return {
        "seed": config.seed,
        "hyperparameters": hyperparameters,
        "config_hash": hashlib.sha256(blob).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
    }


def _rollout(
    env: IADSFormationEnv, learner: QMIXLearner, epsilon: float, seed: int | None
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    horizon = env.config.horizon_t
    obs_buf = np.zeros((horizon, env.n_agents, env.obs_dim), dtype=np.float32)
    act_buf = np.zeros((horizon, env.n_agents), dtype=np.int64)
    rew_buf = np.zeros(horizon, dtype=np.float32)
    state_buf = np.zeros((horizon, env.state_dim), dtype=np.float32)
    done_buf = np.zeros(horizon, dtype=np.float32)
    filled_buf = np.zeros(horizon, dtype=np.float32)

    obs, state, info = env.reset(seed=seed)
    hidden = learner.init_hidden()
    done = False
    t = 0
    while not done and t < horizon:
        actions, hidden = learner.select_actions(obs, hidden, epsilon)
        obs_buf[t] = np.stack([obs[a] for a in range(env.n_agents)])
        act_buf[t] = np.array([actions[a] for a in range(env.n_agents)], dtype=np.int64)
        state_buf[t] = state
        obs, state, rewards, terminated, truncated, info = env.step(actions)
        rew_buf[t] = rewards[0]
        done = terminated or truncated
        done_buf[t] = 1.0 if done else 0.0
        filled_buf[t] = 1.0
        t += 1
    episode = {
        "obs": obs_buf,
        "actions": act_buf,
        "rewards": rew_buf,
        "states": state_buf,
        "dones": done_buf,
        "filled": filled_buf,
    }
    return episode, info


def _evaluate(
    env: IADSFormationEnv, learner: QMIXLearner, n_episodes: int, seed: int
) -> tuple[float, float]:
    wins = 0
    supp_sum = 0.0
    for ep in range(n_episodes):
        _, info = _rollout(env, learner, epsilon=0.0, seed=seed if ep == 0 else None)
        if info["outcome"] == "win":
            wins += 1
        supp_sum += float(info["suppressed_fraction"])
    return wins / n_episodes, supp_sum / n_episodes


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    env = IADSFormationEnv(config.env)
    eval_env = IADSFormationEnv(config.env)
    learner = QMIXLearner(
        env.obs_dim,
        env.action_dim,
        env.n_agents,
        env.state_dim,
        config.agent,
        config.device,
        rng,
    )
    buffer = EpisodeReplayBuffer(
        config.agent.buffer_episodes,
        config.env.horizon_t,
        env.n_agents,
        env.obs_dim,
        env.state_dim,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))
    best_path = out_dir / "best.pt"

    win_rate_history: list[float] = []
    supp_history: list[float] = []
    best_win_rate = -1.0

    for ep in range(config.total_episodes):
        epsilon = _epsilon(ep, config.agent)
        episode, _ = _rollout(env, learner, epsilon, seed=config.seed if ep == 0 else None)
        buffer.add(**episode)
        if len(buffer) >= config.agent.learning_starts_episodes:
            learner.update(buffer.sample(config.agent.batch_episodes, rng))
        if (ep + 1) % config.eval_every == 0:
            win_rate, supp = _evaluate(eval_env, learner, config.eval_episodes, config.seed)
            win_rate_history.append(win_rate)
            supp_history.append(supp)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(learner.agent.state_dict(), best_path)

    if not win_rate_history:
        win_rate, supp = _evaluate(eval_env, learner, config.eval_episodes, config.seed)
        win_rate_history.append(win_rate)
        supp_history.append(supp)
        torch.save(learner.agent.state_dict(), best_path)

    sample = torch.zeros(1, env.obs_dim, dtype=torch.float32)
    mean_ms, p99_ms = profile_latency(
        _AgentForward(learner.agent), sample, n_warmup=5, n_iter=50, device=config.device
    )
    final = {
        "win_rate": win_rate_history[-1],
        "suppressed_fraction": supp_history[-1],
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    return {"win_rate_history": win_rate_history, "final": final}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_train.py -v`
Expected: PASS (todos).

- [ ] **Step 5: Lint, format, type-check, suite completa**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/ tests/marl_formation/ && .venv/bin/ruff check src/cog_ew/marl_formation/ && .venv/bin/mypy src/cog_ew/marl_formation/ && .venv/bin/python -m pytest tests/ -q`
Expected: todo limpio, suite completa en verde.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/train.py configs/marl_formation/qmix.yaml tests/marl_formation/test_train.py
git commit -m "$(cat <<'EOF'
feat(marl): bucle de entrenamiento CTDE (QMIX) + qmix.yaml

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Notas de implementación transversales

- **`_rollout` y `_evaluate` deben quedar consistentes en tipos:** `_rollout` devuelve `(episode, info)` y los llamadores leen `info["outcome"]` y `info["suppressed_fraction"]`. Si mypy se queja del tipo de `info`, anótalo como `dict[str, Any]`.
- **Seeding de episodios:** solo el primer `reset` lleva `seed` explícito; los siguientes usan el RNG interno del env para variar emisores entre episodios (igual patrón que el Modelo 1). El determinismo global lo da `_set_seeds` + el RNG sembrado del learner.
- **Latencia descentralizada:** se perfila `_AgentForward` (un agente, batch=1) porque en ejecución cada aeronave corre solo su `AgentRNN`; el mixer NO interviene en inferencia.
- **mypy `--strict`:** usa `# type: ignore[no-untyped-call]` en `loss.backward()`; tipa los buffers como `NDArray[...]`.
