import importlib

import pytest

MODULES = [
    "cog_ew.deep_rl_jamming.train",
    "cog_ew.temporal_cnn_elint.train",
    "cog_ew.marl_formation.train",
    "cog_ew.gan_signals.train",
    "cog_ew.gan_signals.export",
    "cog_ew.gan_signals.robustness",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_set_seeds_seeds_cuda(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    calls: list[int] = []
    monkeypatch.setattr(module.torch, "manual_seed", lambda s: None)
    monkeypatch.setattr(module.torch.cuda, "manual_seed_all", lambda s: calls.append(s))
    module._set_seeds(123)
    assert calls == [123]
