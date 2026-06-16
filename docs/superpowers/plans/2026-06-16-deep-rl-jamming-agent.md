# Agente D3QN + entrenamiento (Modelo 1, sub-pieza B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un agente Dueling Double DQN (D3QN) en PyTorch y un bucle de entrenamiento reproducible sobre `RadarJammingEnv`, con evaluación por win rate, perfilado de latencia (<5 ms) y logging de reproducibilidad.

**Architecture:** `agent.py` contiene `QNetwork` (MLP sobre obs aplanada + cabezas dueling), `ReplayBuffer` (uniforme, NumPy), `D3QNConfig` y `D3QNAgent` (Double DQN, ε-greedy, Huber, target sync). `train.py` contiene `TrainConfig` (env+agent anidados desde YAML) y `train()` (bucle, evaluación greedy por win rate, mejor checkpoint, `profile_latency`, `run_meta.json`/`metrics.json`).

**Tech Stack:** Python 3.11, PyTorch, NumPy, Gymnasium, PyYAML, pytest, ruff, mypy. Herramientas vía `.venv/bin/<tool>`.

---

## File Structure

- Modify: `src/cog_ew/deep_rl_jamming/agent.py` — `QNetwork`, `ReplayBuffer`, `D3QNConfig`, `D3QNAgent` (sobrescribe el stub, mantiene el docstring).
- Modify: `src/cog_ew/deep_rl_jamming/train.py` — `TrainConfig` + helpers + `train()` (sobrescribe el stub).
- Create: `configs/deep_rl_jamming/train.yaml` — secciones `env_config` (path) + `agent` + entrenamiento.
- Test: `tests/deep_rl_jamming/test_agent.py`, `tests/deep_rl_jamming/test_train.py`.

**Interfaz ya disponible (sub-pieza A, no modificar):** `RadarJammingEnv` (`gymnasium.Env`) con
`observation_space = Box(float32, (8, 5))`, `action_space = Discrete(40)`, `reset(seed=...) -> (obs, info)`,
`step(action) -> (obs, reward, terminated, truncated, info)`, `info["outcome"] ∈ {ongoing, win, lose}`.
`RadarEnvConfig.from_yaml(path)` carga su config. `profile_latency(model, sample, *, n_warmup, n_iter,
device)` está en `cog_ew.temporal_cnn_elint.metrics`.

**Nota de diseño:** `obs_dim` (40 = 8×5) y `n_actions` (40) se derivan del entorno en `train()` y se pasan al
agente; `D3QNConfig` solo guarda hiperparámetros de aprendizaje (no `obs_dim`/`n_actions`), evitando
desajustes con el entorno.

---

## Task 1: `agent.py` — `QNetwork` + `D3QNConfig`

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/agent.py`
- Test: `tests/deep_rl_jamming/test_agent.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/deep_rl_jamming/test_agent.py`:

```python
import torch

from cog_ew.deep_rl_jamming.agent import D3QNConfig, QNetwork


def test_qnetwork_forward_shape():
    net = QNetwork(obs_dim=40, n_actions=40, hidden=16)
    obs = torch.randn(3, 8, 5)
    q = net(obs)
    assert q.shape == (3, 40)


def test_dueling_q_mean_equals_value():
    net = QNetwork(obs_dim=40, n_actions=40, hidden=16)
    obs = torch.randn(4, 8, 5)
    q = net(obs)
    h = net.trunk(obs.flatten(start_dim=1))
    v = net.value(h).squeeze(1)
    assert torch.allclose(q.mean(dim=1), v, atol=1e-5)


def test_d3qn_config_defaults():
    config = D3QNConfig()
    assert config.hidden == 128
    assert config.gamma == 0.99
    assert config.target_sync == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_agent.py -v`
Expected: FAIL con `cannot import name 'QNetwork'`.

- [ ] **Step 3: Write minimal implementation**

Sobrescribir `src/cog_ew/deep_rl_jamming/agent.py` (mantener el docstring del módulo) con:

```python
"""Agente Deep RL para generación de técnicas de jamming adaptativas (<5ms latencia)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs.flatten(start_dim=1))
        value = self.value(h)
        advantage = self.advantage(h)
        q: torch.Tensor = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


@dataclass(frozen=True)
class D3QNConfig:
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    target_sync: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    learning_starts: int = 1000
    train_freq: int = 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/agent.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
git commit -m "feat(deep-rl): QNetwork dueling + D3QNConfig"
```

---

## Task 2: `agent.py` — `ReplayBuffer`

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/agent.py`
- Test: `tests/deep_rl_jamming/test_agent.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_agent.py` (añadir `import numpy as np` arriba y `ReplayBuffer` al import de `agent`):

```python
import numpy as np


def _fill_buffer(buf, n):
    rng = np.random.default_rng(0)
    for _ in range(n):
        obs = rng.standard_normal((8, 5)).astype(np.float32)
        nxt = rng.standard_normal((8, 5)).astype(np.float32)
        buf.add(obs, int(rng.integers(40)), float(rng.standard_normal()), nxt, False)


def test_replay_buffer_len_caps_at_capacity():
    from cog_ew.deep_rl_jamming.agent import ReplayBuffer

    buf = ReplayBuffer(capacity=10, obs_shape=(8, 5))
    _fill_buffer(buf, 25)
    assert len(buf) == 10


def test_replay_buffer_sample_shapes():
    from cog_ew.deep_rl_jamming.agent import ReplayBuffer

    buf = ReplayBuffer(capacity=100, obs_shape=(8, 5))
    _fill_buffer(buf, 50)
    obs, actions, rewards, next_obs, dones = buf.sample(16, np.random.default_rng(0))
    assert obs.shape == (16, 8, 5)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_obs.shape == (16, 8, 5)
    assert dones.shape == (16,)


def test_replay_buffer_sample_is_deterministic_by_rng():
    from cog_ew.deep_rl_jamming.agent import ReplayBuffer

    buf = ReplayBuffer(capacity=100, obs_shape=(8, 5))
    _fill_buffer(buf, 50)
    a = buf.sample(8, np.random.default_rng(3))
    b = buf.sample(8, np.random.default_rng(3))
    assert np.array_equal(a[1], b[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_agent.py -v`
Expected: FAIL con `cannot import name 'ReplayBuffer'`.

- [ ] **Step 3: Write minimal implementation**

Añadir a `src/cog_ew/deep_rl_jamming/agent.py` (añadir `import numpy as np` y `from numpy.typing import NDArray` arriba):

```python
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._pos = 0

    def add(
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        done: bool,
    ) -> None:
        i = self._pos
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        idx = rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_agent.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/agent.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
git commit -m "feat(deep-rl): ReplayBuffer uniforme"
```

---

## Task 3: `agent.py` — `D3QNAgent` (select_action + update)

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/agent.py`
- Test: `tests/deep_rl_jamming/test_agent.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_agent.py` (añadir `D3QNAgent` al import de `agent`):

```python
def _agent(seed=0):
    from cog_ew.deep_rl_jamming.agent import D3QNAgent, D3QNConfig

    return D3QNAgent(
        obs_dim=40,
        n_actions=40,
        config=D3QNConfig(hidden=16, target_sync=100000),
        device="cpu",
        rng=np.random.default_rng(seed),
    )


def test_select_action_greedy_is_argmax():
    agent = _agent()
    obs = np.random.default_rng(1).standard_normal((8, 5)).astype(np.float32)
    with torch.no_grad():
        q = agent.online_net(torch.as_tensor(obs).unsqueeze(0))
    assert agent.select_action(obs, epsilon=0.0) == int(q.argmax())


def test_select_action_random_is_in_range_and_deterministic():
    obs = np.zeros((8, 5), dtype=np.float32)
    a = _agent(5)
    b = _agent(5)
    acts_a = [a.select_action(obs, epsilon=1.0) for _ in range(5)]
    acts_b = [b.select_action(obs, epsilon=1.0) for _ in range(5)]
    assert acts_a == acts_b
    assert all(0 <= x < 40 for x in acts_a)


def test_update_reduces_loss_on_fixed_batch():
    agent = _agent()
    gen = np.random.default_rng(1)
    batch = (
        gen.standard_normal((32, 8, 5)).astype(np.float32),
        gen.integers(0, 40, 32),
        gen.standard_normal(32).astype(np.float32),
        gen.standard_normal((32, 8, 5)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
    )
    first = agent.update(batch)
    last = first
    for _ in range(60):
        last = agent.update(batch)
    assert last < first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_agent.py -v`
Expected: FAIL con `cannot import name 'D3QNAgent'`.

- [ ] **Step 3: Write minimal implementation**

Añadir a `src/cog_ew/deep_rl_jamming/agent.py` (añadir `from torch.nn import functional as F` arriba):

```python
class D3QNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: D3QNConfig,
        device: str,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.rng = rng
        self.online_net = QNetwork(obs_dim, n_actions, config.hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, config.hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self._updates = 0

    @torch.no_grad()
    def select_action(self, obs: NDArray[np.float32], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(torch.argmax(self.online_net(tensor), dim=1).item())

    def update(
        self,
        batch: tuple[
            NDArray[np.float32],
            NDArray[np.int64],
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.float32],
        ],
    ) -> float:
        obs, actions, rewards, next_obs, dones = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q = self.online_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = torch.argmax(self.online_net(next_t), dim=1)
            next_q = self.target_net(next_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.config.target_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_agent.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/agent.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/agent.py tests/deep_rl_jamming/test_agent.py
git commit -m "feat(deep-rl): D3QNAgent (Double DQN, ε-greedy, Huber, target sync)"
```

---

## Task 4: `train.py` — `TrainConfig` + `train.yaml`

**Files:**
- Create: `configs/deep_rl_jamming/train.yaml`
- Modify: `src/cog_ew/deep_rl_jamming/train.py`
- Test: `tests/deep_rl_jamming/test_train.py`

- [ ] **Step 1: Write the failing test**

Crear `tests/deep_rl_jamming/test_train.py`:

```python
from cog_ew.deep_rl_jamming.agent import D3QNConfig
from cog_ew.deep_rl_jamming.env import RadarEnvConfig
from cog_ew.deep_rl_jamming.train import TrainConfig

CONFIG = "configs/deep_rl_jamming/train.yaml"


def test_train_config_from_yaml_parses_nested_sections():
    config = TrainConfig.from_yaml(CONFIG)
    assert isinstance(config.env, RadarEnvConfig)
    assert isinstance(config.agent, D3QNConfig)
    assert config.env.history_k == 8
    assert config.agent.hidden == 128
    assert config.total_steps > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_train.py -v`
Expected: FAIL con `cannot import name 'TrainConfig'`.

- [ ] **Step 3: Crear `configs/deep_rl_jamming/train.yaml`**

```yaml
env_config: configs/deep_rl_jamming/env.yaml
agent:
  hidden: 128
  gamma: 0.99
  lr: 0.001
  batch_size: 64
  buffer_size: 50000
  target_sync: 500
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 5000
  learning_starts: 1000
  train_freq: 1
total_steps: 20000
eval_episodes: 50
eval_every: 2000
device: cpu
seed: 0
out_dir: runs/deep_rl_jamming
tracking: false
```

- [ ] **Step 4: Write minimal implementation**

Sobrescribir `src/cog_ew/deep_rl_jamming/train.py` (mantener el docstring) con el `TrainConfig` (el resto se añade en la Task 5):

```python
"""Script de entrenamiento del agente Deep RL de jamming adaptativo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cog_ew.deep_rl_jamming.agent import D3QNConfig
from cog_ew.deep_rl_jamming.env import RadarEnvConfig


@dataclass
class TrainConfig:
    env: RadarEnvConfig
    agent: D3QNConfig
    total_steps: int = 20000
    eval_episodes: int = 50
    eval_every: int = 2000
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs/deep_rl_jamming"
    tracking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        env = RadarEnvConfig.from_yaml(raw.pop("env_config"))
        agent = D3QNConfig(**raw.pop("agent"))
        return cls(env=env, agent=agent, **raw)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_train.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/train.py tests/deep_rl_jamming/test_train.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/train.py tests/deep_rl_jamming/test_train.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/train.py
```
Expected: sin errores.

- [ ] **Step 7: Commit**

```bash
git add configs/deep_rl_jamming/train.yaml src/cog_ew/deep_rl_jamming/train.py tests/deep_rl_jamming/test_train.py
git commit -m "feat(deep-rl): TrainConfig anidada (env+agent) desde YAML"
```

---

## Task 5: `train.py` — `train()` (bucle + evaluación + latencia + metadata)

**Files:**
- Modify: `src/cog_ew/deep_rl_jamming/train.py`
- Test: `tests/deep_rl_jamming/test_train.py`

- [ ] **Step 1: Write the failing test**

Añadir a `tests/deep_rl_jamming/test_train.py` (añadir `import json` arriba y `train` al import de `train`):

```python
import json


def _tiny_config(out_dir):
    env = RadarEnvConfig.from_yaml("configs/deep_rl_jamming/env.yaml")
    agent = D3QNConfig(
        hidden=16,
        buffer_size=500,
        batch_size=16,
        learning_starts=32,
        target_sync=100,
        epsilon_decay_steps=100,
    )
    return TrainConfig(
        env=env,
        agent=agent,
        total_steps=200,
        eval_episodes=3,
        eval_every=100,
        device="cpu",
        seed=0,
        out_dir=str(out_dir),
        tracking=False,
    )


def test_train_smoke_writes_artifacts_and_metrics(tmp_path):
    from cog_ew.deep_rl_jamming.train import train

    result = train(_tiny_config(tmp_path))

    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "run_meta.json").exists()
    assert len(result["win_rate_history"]) >= 1
    assert result["final"]["latency_mean_ms"] > 0.0

    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["seed"] == 0
    assert "torch" in meta["dependencies"]
    assert "gymnasium" in meta["dependencies"]


def test_train_is_deterministic(tmp_path):
    from cog_ew.deep_rl_jamming.train import train

    a = train(_tiny_config(tmp_path / "a"))
    b = train(_tiny_config(tmp_path / "b"))
    assert a["win_rate_history"] == b["win_rate_history"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_train.py -v`
Expected: FAIL con `cannot import name 'train'`.

- [ ] **Step 3: Write minimal implementation**

Reemplazar los imports y añadir helpers + `train()` en `src/cog_ew/deep_rl_jamming/train.py`. Los imports
completos del fichero quedan así (sustituyen al bloque de imports de la Task 4):

```python
"""Script de entrenamiento del agente Deep RL de jamming adaptativo."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
import yaml

from cog_ew.deep_rl_jamming.agent import D3QNAgent, D3QNConfig, ReplayBuffer
from cog_ew.deep_rl_jamming.env import RadarEnvConfig, RadarJammingEnv
from cog_ew.temporal_cnn_elint.metrics import profile_latency
```

(El `TrainConfig` de la Task 4 se mantiene tal cual debajo de los imports.)

Añadir al final del fichero:

```python
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epsilon(step: int, config: D3QNConfig) -> float:
    frac = min(1.0, step / config.epsilon_decay_steps)
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


def _evaluate(env: RadarJammingEnv, agent: D3QNAgent, n_episodes: int, seed: int) -> float:
    wins = 0
    obs, info = env.reset(seed=seed)
    for _ in range(n_episodes):
        done = False
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info["outcome"] == "win":
            wins += 1
        obs, info = env.reset()
    return wins / n_episodes


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    env = RadarJammingEnv(config.env)
    eval_env = RadarJammingEnv(config.env)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    obs_dim = int(np.prod(obs_shape))
    n_actions = int(env.action_space.n)

    agent = D3QNAgent(obs_dim, n_actions, config.agent, config.device, rng)
    buffer = ReplayBuffer(config.agent.buffer_size, obs_shape)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(_run_metadata(config), indent=2))
    best_path = out_dir / "best.pt"

    win_rate_history: list[float] = []
    best_win_rate = -1.0

    obs, _ = env.reset(seed=config.seed)
    for step in range(config.total_steps):
        action = agent.select_action(obs, _epsilon(step, config.agent))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if len(buffer) >= config.agent.learning_starts and step % config.agent.train_freq == 0:
            agent.update(buffer.sample(config.agent.batch_size, rng))

        if (step + 1) % config.eval_every == 0:
            win_rate = _evaluate(eval_env, agent, config.eval_episodes, config.seed)
            win_rate_history.append(win_rate)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(agent.online_net.state_dict(), best_path)

    if not win_rate_history:
        win_rate = _evaluate(eval_env, agent, config.eval_episodes, config.seed)
        win_rate_history.append(win_rate)
        torch.save(agent.online_net.state_dict(), best_path)

    sample = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mean_ms, p99_ms = profile_latency(
        agent.online_net, sample, n_warmup=5, n_iter=50, device=config.device
    )
    final = {
        "win_rate": win_rate_history[-1],
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    return {"win_rate_history": win_rate_history, "final": final}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/deep_rl_jamming/test_train.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, format, types**

```bash
.venv/bin/ruff format src/cog_ew/deep_rl_jamming/train.py tests/deep_rl_jamming/test_train.py
.venv/bin/ruff check src/cog_ew/deep_rl_jamming/train.py tests/deep_rl_jamming/test_train.py
.venv/bin/mypy src/cog_ew/deep_rl_jamming/train.py
```
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/deep_rl_jamming/train.py tests/deep_rl_jamming/test_train.py
git commit -m "feat(deep-rl): bucle D3QN + eval por win rate + latencia + run_meta"
```

---

## Task 6: Verificación final (lint, tipos, suite completa)

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `.venv/bin/python -m pytest -q`
Expected: todos los tests pasan (los anteriores + los nuevos de agente/train).

- [ ] **Step 2: Lint y formato**

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```
Expected: `All checks passed!` y todos los ficheros formateados.

- [ ] **Step 3: Tipos sobre `src/`**

Run: `.venv/bin/mypy src`
Expected: `Success: no issues found`.

- [ ] **Step 4: Commit (si hubo algún ajuste)**

```bash
git add -A && git commit -m "chore(deep-rl): verificación final del agente D3QN" || echo "nada que commitear"
```

---

## Self-Review (cobertura del spec)

- **`QNetwork` MLP + dueling:** Task 1; test `test_dueling_q_mean_equals_value` verifica `Q = V + (A − mean A)`. ✅
- **`ReplayBuffer` uniforme:** Task 2; shapes + determinismo por rng. ✅
- **`D3QNConfig`:** Tasks 1/4; hiperparámetros de aprendizaje (sin `obs_dim`/`n_actions`, derivados del env). ✅
- **`D3QNAgent` (Double DQN, ε-greedy, Huber, target sync):** Task 3; greedy=argmax, random determinista, update reduce pérdida. ✅
- **`TrainConfig` env+agent anidados desde YAML:** Task 4; `from_yaml` con `env_config` (path) + `agent`. ✅
- **Bucle + eval por win rate + mejor checkpoint:** Task 5; `_evaluate` greedy, `best.pt` por win rate. ✅
- **Perfilado de latencia (<5 ms):** Task 5; `profile_latency` sobre `online_net`, en `metrics.json`. ✅
- **Reproducibilidad (seeds, run_meta incondicional, trackio opcional):** Task 5; `_set_seeds`, `run_meta.json` con deps (incl. gymnasium), determinismo testeado. ✅
- **Fuera de alcance respetado:** sin comparación vs baseline (C), sin replay priorizado, sin Colab/>92 %. ✅
- **Consistencia de tipos:** `D3QNConfig`/`RadarEnvConfig` usados por `TrainConfig` (Task 4) y `train` (Task 5); `D3QNAgent(obs_dim, n_actions, config, device, rng)` idéntico entre Task 3 y su uso en Task 5; `ReplayBuffer(capacity, obs_shape)` y `.sample(batch, rng)` consistentes entre Tasks 2 y 5. ✅

**Nota:** el smoke test entrena 200 pasos con red de 16 unidades (segundos en CPU); no verifica el >92 % (Fase 6), solo que el bucle corre, aprende, vuelca artefactos y es determinista.
