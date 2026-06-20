import numpy as np
import torch

from cog_ew.marl_formation.agents import AgentRNN, QMIXConfig, QMixer, QMIXLearner


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
