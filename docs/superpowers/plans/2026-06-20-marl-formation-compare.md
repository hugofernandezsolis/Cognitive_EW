# Comparación QMIX vs IQL (Modelo 3, sub-pieza C) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir la mejora de supresión del IADS de la coordinación QMIX frente a learners independientes (IQL), anclando el +45 % de la Propuesta.

**Architecture:** `IQLLearner` reutiliza el `AgentRNN` compartido y el `EpisodeReplayBuffer` de la sub-pieza B, pero entrena cada agente con su propia pérdida Double-DQN sobre la recompensa de equipo, sin `QMixer` ni estado global. El bucle de entrenamiento se generaliza a dos regímenes (`qmix`/`iql`). En ejecución ambos regímenes corren igual (AgentRNN greedy descentralizado), así que `compare.py` compara dos checkpoints y reporta la mejora relativa de `suppressed_fraction`.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML. Tests con pytest. Reutiliza `AgentRNN`, `QMIXConfig`, `EpisodeReplayBuffer`, `IADSFormationEnv`, `profile_latency`.

## Global Constraints

- **Type hints** en todas las funciones públicas; mypy `--strict` debe pasar (`loss.backward()  # type: ignore[no-untyped-call]`).
- **ruff check** y **ruff format** limpios antes de cada commit (line-length 100).
- **Reproducibilidad:** seeds explícitos; hiperparámetros SOLO en YAML versionado; `run_meta.json` con `regime` + seed + hyperparams + hash + versiones.
- **Seguridad EW:** no exponer parámetros de amenazas reales; `torch.load(..., weights_only=True)` al recargar checkpoints.
- **Herramientas:** tests con `.venv/bin/python -m pytest`; ruff/mypy con `.venv/bin/ruff` y `.venv/bin/mypy`.
- **Commits** terminan con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Ejecución idéntica entre regímenes:** `IQLLearner` expone `init_hidden`/`select_actions` con el MISMO contrato que `QMIXLearner` (ε-greedy descentralizada por agente).
- **IQL no usa estado global:** su constructor no recibe `state_dim`; su `update` ignora `states` del batch.
- **Dimensiones del entorno** (`env.yaml`: M=4, N=4, P=4): `obs_dim=24`, `state_dim=28`, `action_dim=48`.

---

### Task 1: `IQLLearner` — base compartida de ejecución + constructor + `select_actions`

**Files:**
- Modify: `src/cog_ew/marl_formation/agents.py`
- Test: `tests/marl_formation/test_agents.py`

**Interfaces:**
- Consumes: `AgentRNN`, `QMIXConfig` (ya existen).
- Produces:
  - `_SharedParamLearner` (base) con `init_hidden() -> dict[int, Tensor]` y `select_actions(obs_dict, hidden, epsilon) -> tuple[dict[int,int], dict[int,Tensor]]`.
  - `IQLLearner(obs_dim, action_dim, n_agents, config, device, rng)` que hereda de `_SharedParamLearner`; expone `self.agent: AgentRNN`, `self.target_agent`, `self.optimizer`. (Método `update` en la Task 2.)
  - `QMIXLearner` pasa a heredar de `_SharedParamLearner` (sus `init_hidden`/`select_actions` se eliminan y se heredan).

- [ ] **Step 1: Write the failing test**

```python
def _iql() -> "QMIXLearner | IQLLearner":
    from cog_ew.marl_formation.agents import IQLLearner, QMIXConfig

    return IQLLearner(
        obs_dim=24,
        action_dim=48,
        n_agents=4,
        config=QMIXConfig(hidden=16),
        device="cpu",
        rng=np.random.default_rng(0),
    )


def test_iql_select_actions_returns_valid_dict():
    learner = _iql()
    obs = {a: np.zeros(24, dtype=np.float32) for a in range(4)}
    actions, hidden = learner.select_actions(obs, learner.init_hidden(), epsilon=0.0)
    assert set(actions) == {0, 1, 2, 3}
    assert all(0 <= actions[a] < 48 for a in range(4))
    assert set(hidden) == {0, 1, 2, 3}


def test_iql_select_actions_greedy_is_deterministic():
    learner = _iql()
    obs = {a: np.ones(24, dtype=np.float32) for a in range(4)}
    a1, _ = learner.select_actions(obs, learner.init_hidden(), epsilon=0.0)
    a2, _ = learner.select_actions(obs, learner.init_hidden(), epsilon=0.0)
    assert a1 == a2


def test_iql_has_no_mixer():
    learner = _iql()
    assert not hasattr(learner, "mixer")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k iql -v`
Expected: FAIL with `ImportError: cannot import name 'IQLLearner'`.

- [ ] **Step 3: Write minimal implementation**

En `agents.py`, añade la base ANTES de `QMIXLearner` (tras `QMIXConfig`):

```python
class _SharedParamLearner:
    """Ejecución descentralizada compartida: AgentRNN por agente + ε-greedy."""

    agent: AgentRNN
    n_agents: int
    action_dim: int
    rng: np.random.Generator
    device: torch.device

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
```

Cambia la declaración de `QMIXLearner` a `class QMIXLearner(_SharedParamLearner):` y **elimina** sus métodos `init_hidden` y `select_actions` (ahora heredados). No cambies nada más de `QMIXLearner`.

Añade `IQLLearner` tras `QMIXLearner`:

```python
class IQLLearner(_SharedParamLearner):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
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
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.lr)
        self._updates = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -v`
Expected: PASS (los nuevos de IQL + todos los de QMIX siguen verdes — confirma que el refactor de la base no rompe QMIX).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py && .venv/bin/ruff check src/cog_ew/marl_formation/agents.py && .venv/bin/mypy src/cog_ew/marl_formation/agents.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py
git commit -m "$(cat <<'EOF'
feat(marl): IQLLearner + base de ejecución compartida (learners independientes)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `IQLLearner.update` — Double-DQN por agente, sin mixer

**Files:**
- Modify: `src/cog_ew/marl_formation/agents.py`
- Test: `tests/marl_formation/test_agents.py`

**Interfaces:**
- Consumes: `IQLLearner` (Task 1).
- Produces: `IQLLearner.update(batch) -> float`, con `batch` del mismo formato que `QMIXLearner.update`:
  `(obs (B,T,N,obs_dim), actions (B,T,N) int64, rewards (B,T), states (B,T,state_dim), dones (B,T), filled (B,T))`. `states` se ignora.

- [ ] **Step 1: Write the failing test**

```python
def _iql_batch(B=2, T=5, N=4, obs_dim=24, state_dim=28, action_dim=48):
    rng = np.random.default_rng(2)
    obs = rng.standard_normal((B, T, N, obs_dim)).astype(np.float32)
    actions = rng.integers(0, action_dim, size=(B, T, N)).astype(np.int64)
    rewards = rng.standard_normal((B, T)).astype(np.float32)
    states = rng.standard_normal((B, T, state_dim)).astype(np.float32)
    dones = np.zeros((B, T), dtype=np.float32)
    dones[:, -1] = 1.0
    filled = np.ones((B, T), dtype=np.float32)
    return obs, actions, rewards, states, dones, filled


def test_iql_update_returns_finite_loss_and_changes_params():
    learner = _iql()
    before = next(learner.agent.parameters()).clone()
    loss = learner.update(_iql_batch())
    after = next(learner.agent.parameters())
    assert np.isfinite(loss)
    assert not torch.allclose(before, after)


def test_iql_update_syncs_target_after_interval():
    from cog_ew.marl_formation.agents import IQLLearner, QMIXConfig

    config = QMIXConfig(hidden=16, target_sync=1)
    learner = IQLLearner(24, 48, 4, config, "cpu", np.random.default_rng(0))
    learner.update(_iql_batch())
    for online, target in zip(
        learner.agent.parameters(), learner.target_agent.parameters(), strict=True
    ):
        assert torch.allclose(online, target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -k "iql_update" -v`
Expected: FAIL with `AttributeError: 'IQLLearner' object has no attribute 'update'`.

- [ ] **Step 3: Write minimal implementation**

Añade el método `update` a `IQLLearner`:

```python
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
        obs_np, actions_np, rewards_np, _states_np, dones_np, filled_np = batch
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
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

        with torch.no_grad():
            reward_b = rewards[:, :-1].unsqueeze(2)
            done_b = dones[:, :-1].unsqueeze(2)
            y = reward_b + self.config.gamma * (1.0 - done_b) * target_max[:, 1:]

        td_error = F.smooth_l1_loss(chosen[:, :-1], y, reduction="none")
        mask = filled[:, :-1].unsqueeze(2).expand_as(td_error)
        loss = (td_error * mask).sum() / mask.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.config.target_sync == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
        return float(loss.item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_agents.py -v`
Expected: PASS (todos los de agents, QMIX + IQL).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py && .venv/bin/ruff check src/cog_ew/marl_formation/agents.py && .venv/bin/mypy src/cog_ew/marl_formation/agents.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/agents.py tests/marl_formation/test_agents.py
git commit -m "$(cat <<'EOF'
feat(marl): IQLLearner.update (Double-DQN por agente, sin mezcla centralizada)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Bucle de entrenamiento con régimen `qmix`/`iql` + `iql.yaml`

**Files:**
- Modify: `src/cog_ew/marl_formation/train.py`
- Modify: `configs/marl_formation/qmix.yaml`
- Create: `configs/marl_formation/iql.yaml`
- Test: `tests/marl_formation/test_train.py`

**Interfaces:**
- Consumes: `QMIXLearner`, `IQLLearner` (Tasks 1-2), `IADSFormationEnv`.
- Produces:
  - `TrainConfig.regime: str = "qmix"` (campo nuevo; `from_yaml` lo lee).
  - `_build_learner(regime, env, config, device, rng) -> QMIXLearner | IQLLearner`.
  - `train(config)` usa `_build_learner(config.regime, ...)`; `run_meta.json` incluye `regime`.

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import replace

from cog_ew.marl_formation.agents import QMIXConfig


def test_train_config_regime_defaults_to_qmix():
    config = TrainConfig.from_yaml("configs/marl_formation/qmix.yaml")
    assert config.regime == "qmix"


def test_train_config_iql_yaml_sets_regime():
    config = TrainConfig.from_yaml("configs/marl_formation/iql.yaml")
    assert config.regime == "iql"


def _smoke(config: TrainConfig, out_dir) -> TrainConfig:
    agent = replace(config.agent, learning_starts_episodes=2, batch_episodes=2)
    return replace(
        config,
        agent=agent,
        total_episodes=6,
        eval_episodes=3,
        eval_every=3,
        out_dir=str(out_dir),
    )


def test_train_iql_regime_smoke(tmp_path):
    config = _smoke(TrainConfig.from_yaml("configs/marl_formation/iql.yaml"), tmp_path)
    result = train(config)
    assert 0.0 <= result["final"]["win_rate"] <= 1.0
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_train_iql_is_deterministic_by_seed(tmp_path):
    base = TrainConfig.from_yaml("configs/marl_formation/iql.yaml")
    a = train(_smoke(base, tmp_path / "a"))
    b = train(_smoke(base, tmp_path / "b"))
    assert a["win_rate_history"] == b["win_rate_history"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_train.py -k "regime or iql" -v`
Expected: FAIL — `iql.yaml` no existe / `TrainConfig` no tiene `regime`.

- [ ] **Step 3a: Create `configs/marl_formation/iql.yaml`**

```yaml
env_config: configs/marl_formation/env.yaml
regime: iql
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
out_dir: runs/marl_formation/iql
tracking: false
```

- [ ] **Step 3b: Add `regime: qmix` to `configs/marl_formation/qmix.yaml`**

Inserta `regime: qmix` justo después de la línea `env_config: configs/marl_formation/env.yaml` (línea 1) y cambia `out_dir: runs/marl_formation` → `out_dir: runs/marl_formation/qmix`.

- [ ] **Step 3c: Modify `train.py`**

Añade `IQLLearner` al import existente de agents:

```python
from cog_ew.marl_formation.agents import AgentRNN, IQLLearner, QMIXConfig, QMIXLearner
```

Añade el campo `regime` a `TrainConfig` (tras `tracking: bool = False`):

```python
    regime: str = "qmix"
```

Añade la fábrica tras `TrainConfig` (antes de `_AgentForward`):

```python
def _build_learner(
    regime: str,
    env: IADSFormationEnv,
    config: QMIXConfig,
    device: str,
    rng: np.random.Generator,
) -> QMIXLearner | IQLLearner:
    if regime == "qmix":
        return QMIXLearner(
            env.obs_dim, env.action_dim, env.n_agents, env.state_dim, config, device, rng
        )
    if regime == "iql":
        return IQLLearner(env.obs_dim, env.action_dim, env.n_agents, config, device, rng)
    raise ValueError(f"unknown regime: {regime!r}")
```

En `train`, sustituye el bloque que construye `learner = QMIXLearner(...)` por:

```python
    learner = _build_learner(config.regime, env, config.agent, config.device, rng)
```

(El resto del bucle no cambia: ya opera contra `init_hidden`/`select_actions`/`update`/`.agent`, comunes a ambos. `run_meta.json` ya serializa `regime` vía `asdict(config)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_train.py -v`
Expected: PASS (regímenes qmix + iql; los tests previos de qmix siguen verdes).

- [ ] **Step 5: Lint, format, type-check**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/train.py tests/marl_formation/test_train.py && .venv/bin/ruff check src/cog_ew/marl_formation/train.py && .venv/bin/mypy src/cog_ew/marl_formation/train.py`
Expected: todo limpio.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/train.py configs/marl_formation/qmix.yaml configs/marl_formation/iql.yaml tests/marl_formation/test_train.py
git commit -m "$(cat <<'EOF'
feat(marl): entrenamiento con régimen qmix/iql + iql.yaml

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `compare.py` — `AgentPolicy`, baselines, `evaluate_policy`, `compare_policies`

**Files:**
- Modify (reescritura completa): `src/cog_ew/marl_formation/compare.py`
- Modify (reescritura completa): `tests/marl_formation/test_compare.py`

**Interfaces:**
- Consumes: `AgentRNN` (`agents.py`), `IADSFormationEnv` (`env.py`).
- Produces:
  - `FormationPolicy` (Protocol): `reset(env)` + `act(env, obs, state, info) -> dict[int,int]`.
  - `ConcentratedSuppressionPolicy(target=0)`, `SpreadSuppressionPolicy` (baselines rule-based deterministas).
  - `AgentPolicy(agent, n_agents, device="cpu")` + `AgentPolicy.from_checkpoint(path, *, obs_dim, action_dim, hidden, n_agents, device="cpu")`.
  - `evaluate_policy(env, policy, *, episodes, seed) -> dict[str,float]` (`win_rate`, `mean_reward`, `mean_steps`, `suppressed_fraction`).
  - `compare_policies(env, *, coordinated, independent, episodes, seed) -> dict` con claves `coordinated`, `independent`, `delta`, `relative_improvement`.

- [ ] **Step 1: Write the failing test**

Reescribe `tests/marl_formation/test_compare.py` completo:

```python
import numpy as np
import torch

from cog_ew.marl_formation.agents import AgentRNN
from cog_ew.marl_formation.compare import (
    AgentPolicy,
    ConcentratedSuppressionPolicy,
    SpreadSuppressionPolicy,
    compare_policies,
    evaluate_policy,
)
from cog_ew.marl_formation.env import IADSEnvConfig, IADSFormationEnv

CONFIG = "configs/marl_formation/env.yaml"


def _env() -> IADSFormationEnv:
    return IADSFormationEnv(IADSEnvConfig.from_yaml(CONFIG))


def test_concentrated_policy_points_every_agent_to_same_radar():
    env = _env()
    policy = ConcentratedSuppressionPolicy()
    obs, state, info = env.reset(seed=0)
    actions = policy.act(env, obs, state, info)
    expected = env.encode_action(target=0, jam_type=2, power_level=len(env.config.power_levels) - 1)
    assert actions == {agent: expected for agent in range(env.n_agents)}


def test_spread_policy_distributes_targets_round_robin():
    env = _env()
    policy = SpreadSuppressionPolicy()
    obs, state, info = env.reset(seed=0)
    actions = policy.act(env, obs, state, info)
    decoded = [env._decode_action(actions[agent]) for agent in range(env.n_agents)]
    assert decoded == [(0, 2, 3), (1, 2, 3), (2, 2, 3), (3, 2, 3)]


def test_agent_policy_returns_valid_actions():
    env = _env()
    agent = AgentRNN(env.obs_dim, env.action_dim, hidden=16)
    policy = AgentPolicy(agent=agent, n_agents=env.n_agents, device="cpu")
    obs, state, info = env.reset(seed=0)
    policy.reset(env)
    actions = policy.act(env, obs, state, info)
    assert set(actions) == set(range(env.n_agents))
    assert all(0 <= action < env.action_dim for action in actions.values())


def test_agent_policy_loads_weights_from_checkpoint(tmp_path):
    env = _env()
    agent = AgentRNN(env.obs_dim, env.action_dim, hidden=16)
    path = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), path)
    policy = AgentPolicy.from_checkpoint(
        path,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden=16,
        n_agents=env.n_agents,
        device="cpu",
    )
    obs, state, info = env.reset(seed=0)
    policy.reset(env)
    assert set(policy.act(env, obs, state, info)) == set(range(env.n_agents))


def test_evaluate_policy_is_deterministic_by_seed():
    a = evaluate_policy(_env(), SpreadSuppressionPolicy(), episodes=5, seed=0)
    b = evaluate_policy(_env(), SpreadSuppressionPolicy(), episodes=5, seed=0)
    assert a == b
    assert 0.0 <= a["win_rate"] <= 1.0
    assert np.isfinite(a["mean_reward"])


def test_spread_covers_at_least_concentrated():
    spread = evaluate_policy(_env(), SpreadSuppressionPolicy(), episodes=5, seed=0)
    concentrated = evaluate_policy(_env(), ConcentratedSuppressionPolicy(), episodes=5, seed=0)
    assert spread["suppressed_fraction"] >= concentrated["suppressed_fraction"]
    assert spread["win_rate"] >= concentrated["win_rate"]


def test_compare_policies_reports_absolute_and_relative():
    result = compare_policies(
        _env(),
        coordinated=SpreadSuppressionPolicy(),
        independent=ConcentratedSuppressionPolicy(),
        episodes=5,
        seed=0,
    )
    assert set(result) == {"coordinated", "independent", "delta", "relative_improvement"}
    assert result["delta"]["suppressed_fraction"] == (
        result["coordinated"]["suppressed_fraction"] - result["independent"]["suppressed_fraction"]
    )
    indep = result["independent"]["suppressed_fraction"]
    coord = result["coordinated"]["suppressed_fraction"]
    expected_rel = (coord - indep) / indep if indep > 0 else float("inf")
    assert result["relative_improvement"]["suppressed_fraction"] == expected_rel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_compare.py -v`
Expected: FAIL — `cannot import name 'AgentPolicy'` (el `compare.py` actual expone `QMIXPolicy`, no `AgentPolicy` ni `relative_improvement`).

- [ ] **Step 3: Write minimal implementation**

Reescribe `src/cog_ew/marl_formation/compare.py` completo:

```python
"""Comparación de regímenes coordinado (QMIX) vs independiente (IQL) en el entorno IADS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from cog_ew.marl_formation.agents import AgentRNN
from cog_ew.marl_formation.env import IADSFormationEnv


class FormationPolicy(Protocol):
    def reset(self, env: IADSFormationEnv) -> None: ...

    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]: ...


def _suppress_max(env: IADSFormationEnv, target: int) -> int:
    return env.encode_action(target, jam_type=2, power_level=len(env.config.power_levels) - 1)


class ConcentratedSuppressionPolicy:
    """Baseline sin coordinar: todos los agentes apuntan al mismo radar a máxima potencia."""

    def __init__(self, target: int = 0) -> None:
        self.target = target

    def reset(self, env: IADSFormationEnv) -> None:
        return None

    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]:
        action = _suppress_max(env, self.target % env.n_radars)
        return {agent: action for agent in range(env.n_agents)}


class SpreadSuppressionPolicy:
    """Baseline coordinado rule-based: reparte agentes entre radares round-robin."""

    def reset(self, env: IADSFormationEnv) -> None:
        return None

    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]:
        return {
            agent: _suppress_max(env, agent % env.n_radars) for agent in range(env.n_agents)
        }


class AgentPolicy:
    """Ejecuta una política aprendida (QMIX o IQL) vía AgentRNN greedy descentralizado."""

    def __init__(self, agent: AgentRNN, n_agents: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.agent = agent.to(self.device)
        self.n_agents = n_agents
        self._hidden: dict[int, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        obs_dim: int,
        action_dim: int,
        hidden: int,
        n_agents: int,
        device: str = "cpu",
    ) -> AgentPolicy:
        agent = AgentRNN(obs_dim, action_dim, hidden)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)
        return cls(agent=agent, n_agents=n_agents, device=device)

    def reset(self, env: IADSFormationEnv) -> None:
        self._hidden = {
            agent: self.agent.init_hidden(1).to(self.device) for agent in range(self.n_agents)
        }

    @torch.no_grad()
    def act(
        self,
        env: IADSFormationEnv,
        obs: dict[int, NDArray[np.float32]],
        state: NDArray[np.float32],
        info: dict[str, Any],
    ) -> dict[int, int]:
        if not self._hidden:
            self.reset(env)
        actions: dict[int, int] = {}
        new_hidden: dict[int, torch.Tensor] = {}
        for agent in range(self.n_agents):
            obs_t = torch.as_tensor(
                obs[agent], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values, hidden = self.agent(obs_t, self._hidden[agent])
            actions[agent] = int(torch.argmax(q_values, dim=1).item())
            new_hidden[agent] = hidden
        self._hidden = new_hidden
        return actions


def evaluate_policy(
    env: IADSFormationEnv,
    policy: FormationPolicy,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    wins = 0
    total_reward = 0.0
    total_steps = 0
    suppressed_sum = 0.0
    for episode in range(episodes):
        obs, state, info = env.reset(seed=seed if episode == 0 else None)
        policy.reset(env)
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            actions = policy.act(env, obs, state, info)
            obs, state, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += rewards[0]
            steps += 1
            done = terminated or truncated
        if info["outcome"] == "win":
            wins += 1
        total_reward += episode_reward
        total_steps += steps
        suppressed_sum += float(info["suppressed_fraction"])
    return {
        "win_rate": wins / episodes,
        "mean_reward": total_reward / episodes,
        "mean_steps": total_steps / episodes,
        "suppressed_fraction": suppressed_sum / episodes,
    }


def compare_policies(
    env: IADSFormationEnv,
    *,
    coordinated: FormationPolicy,
    independent: FormationPolicy,
    episodes: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    coord = evaluate_policy(env, coordinated, episodes=episodes, seed=seed)
    indep = evaluate_policy(env, independent, episodes=episodes, seed=seed)
    indep_supp = indep["suppressed_fraction"]
    rel_supp = (
        (coord["suppressed_fraction"] - indep_supp) / indep_supp
        if indep_supp > 0
        else float("inf")
    )
    return {
        "coordinated": coord,
        "independent": indep,
        "delta": {
            "win_rate": coord["win_rate"] - indep["win_rate"],
            "mean_reward": coord["mean_reward"] - indep["mean_reward"],
            "suppressed_fraction": coord["suppressed_fraction"] - indep_supp,
        },
        "relative_improvement": {"suppressed_fraction": rel_supp},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/marl_formation/test_compare.py -v`
Expected: PASS (todos).

- [ ] **Step 5: Lint, format, type-check, suite completa**

Run: `.venv/bin/ruff format src/cog_ew/marl_formation/ tests/marl_formation/ && .venv/bin/ruff check src/cog_ew/marl_formation/ && .venv/bin/mypy src/cog_ew/marl_formation/ && .venv/bin/python -m pytest tests/ -q`
Expected: todo limpio, suite completa en verde.

- [ ] **Step 6: Commit**

```bash
git add src/cog_ew/marl_formation/compare.py tests/marl_formation/test_compare.py
git commit -m "$(cat <<'EOF'
feat(marl): compare QMIX vs IQL (AgentPolicy + mejora relativa de supresión)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Notas de implementación transversales

- **`SpreadSuppressionPolicy` asume `n_agents == n_radars`** (round-robin 1:1, como en `env.yaml`). Con N≠M sigue siendo válido (módulo), solo que el reparto no es perfecto; YAGNI cubrirlo más allá.
- **El valor real del +45 %** sale de entrenar ambos regímenes en Colab (Fase 6); estos tests validan la mecánica del arnés, no la cifra.
- **`AgentPolicy` ignora `state`/`info`** en `act` (ejecución descentralizada pura); los parámetros están en la firma por el Protocol `FormationPolicy`.
- **mypy `--strict`:** la unión `QMIXLearner | IQLLearner` que devuelve `_build_learner` es válida porque ambos comparten `init_hidden`/`select_actions`/`update`/`.agent`.
