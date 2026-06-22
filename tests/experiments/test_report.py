from cog_ew.experiments.report import ExperimentProfile


def test_quick_profile_loads_from_yaml():
    profile = ExperimentProfile.from_yaml("configs/experiments/quick.yaml")
    assert profile.name == "quick"
    assert profile.device == "cpu"
    assert profile.jamming_total_steps is not None and profile.jamming_total_steps > 0
    assert profile.elint_epochs is not None
    assert profile.marl_compare_episodes > 0
    assert profile.jamming_config == "configs/deep_rl_jamming/train.yaml"


def test_full_profile_uses_null_for_yaml_durations():
    profile = ExperimentProfile.from_yaml("configs/experiments/full.yaml")
    assert profile.name == "full"
    assert profile.device == "cuda"
    assert profile.jamming_total_steps is None
    assert profile.elint_epochs is None
    assert profile.gan_total_steps is None
    assert profile.jamming_compare_episodes > 0
