import numpy as np
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
