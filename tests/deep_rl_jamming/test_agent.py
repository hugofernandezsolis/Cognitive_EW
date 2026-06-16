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
