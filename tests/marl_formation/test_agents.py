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
