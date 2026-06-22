import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("run_anchors_cli", Path("notebooks/run_anchors.py"))
assert _SPEC is not None and _SPEC.loader is not None
run_anchors_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_anchors_cli)


def test_parse_args_defaults():
    args = run_anchors_cli.parse_args(["--profile", "quick"])
    assert args.profile == "quick"
    assert args.anchors == "all"


def test_resolve_anchors_all():
    assert run_anchors_cli.resolve_anchors("all") == ("jamming", "elint", "marl", "gan")


def test_resolve_anchors_list():
    assert run_anchors_cli.resolve_anchors("elint,marl") == ("elint", "marl")


def test_main_dispatches_to_run_anchors(monkeypatch, tmp_path):
    captured = {}

    def fake_run_anchors(names, profile, out_dir):
        captured["names"] = names
        captured["profile_name"] = profile.name
        captured["out_dir"] = out_dir
        return {"anchors": {}}

    monkeypatch.setattr(run_anchors_cli, "run_anchors", fake_run_anchors)
    run_anchors_cli.main(["--profile", "quick", "--anchors", "elint", "--out-dir", str(tmp_path)])
    assert captured["names"] == ("elint",)
    assert captured["profile_name"] == "quick"
    assert captured["out_dir"] == str(tmp_path)
