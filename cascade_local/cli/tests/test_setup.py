"""`cascade setup`: model choice by RAM, downloads only what's missing, consent before
downloading or installing, config written, and a clear stop when Ollama isn't running.
No live Ollama, network or model needed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import cascade_local.cli.main as cli_main
from cascade_local.cli.main import app, recommend_model, setup_tags

runner = CliRunner()


class _FakeOllama:
    def __init__(self, available: bool = True, local: tuple[str, ...] = ()) -> None:
        self._available = available
        self._local = list(local)

    def is_available(self) -> bool:
        return self._available

    def local_models(self) -> list[str]:
        return self._local


class _FakeProfiler:
    def __init__(self, capable: bool = True) -> None:
        self._capable = capable

    def profile(self) -> dict:
        return {"measured_tok_s": {"q4_K_M": 3.6}, "local_capable": self._capable}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import psutil

    from cascade_local.engine.execution import model_runner

    state = SimpleNamespace(ollama=_FakeOllama(), pulled=[], asked=[], ram_gb=16.0, llamacpp=True,
                            installs=[], profiler=_FakeProfiler())
    monkeypatch.setattr(cli_main, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(model_runner, "OllamaReferenceModelRunner", lambda: state.ollama)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=state.ram_gb * 2**30))
    monkeypatch.setattr(cli_main, "_pull", lambda tag: state.pulled.append(tag) or True)
    monkeypatch.setattr(cli_main, "hardware_profiler", lambda config, runner=None: state.profiler)
    monkeypatch.setattr(cli_main, "ask", lambda query, no_log=False, stream=True: state.asked.append(query))
    monkeypatch.setattr(cli_main, "_llama_cpp_available", lambda: state.llamacpp)

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: state.installs.append(cmd) or SimpleNamespace(returncode=0))
    return state


def _config(state_dir_config: Path) -> dict:
    return json.loads(state_dir_config.read_text(encoding="utf-8"))


def test_model_choice_follows_total_ram() -> None:
    assert recommend_model(32)[0] == "qwen2.5-coder"
    assert recommend_model(12)[0] == "qwen2.5-coder"
    assert recommend_model(11.9)[0] == "qwen2.5:3b-instruct"
    assert recommend_model(8)[0] == "qwen2.5:3b-instruct"
    assert recommend_model(4)[0] == "qwen2.5:1.5b-instruct"


def test_every_choice_brings_the_shared_small_model() -> None:
    assert setup_tags("qwen2.5-coder") == ["qwen2.5-coder:7b-instruct-q4_K_M", "qwen2.5:0.5b-instruct-q4_K_M"]
    assert setup_tags("qwen2.5:3b-instruct") == ["qwen2.5:3b-instruct-q4_K_M", "qwen2.5:0.5b-instruct-q4_K_M"]
    assert setup_tags("llama3.2") == ["llama3.2:3b-instruct-q4_K_M"]  # no small model for this family


def test_setup_downloads_what_is_missing_writes_config_and_tests(env) -> None:
    env.ollama = _FakeOllama(local=("qwen2.5:0.5b-instruct-q4_K_M",))
    result = runner.invoke(app, ["setup", "--yes"])
    assert result.exit_code == 0, result.output
    assert env.pulled == ["qwen2.5-coder:7b-instruct-q4_K_M"]
    config = _config(cli_main.CONFIG_PATH)
    assert config["model"] == "qwen2.5-coder" and config["initialized"] and config["local_capable"]
    assert len(env.asked) == 1
    assert "Cascade is ready" in result.output


def test_setup_asks_before_downloading_and_stops_on_no(env) -> None:
    result = runner.invoke(app, ["setup"], input="n\n")
    assert result.exit_code == 1
    assert env.pulled == []
    assert "Download" in result.output


def test_setup_with_everything_present_downloads_nothing(env) -> None:
    env.ollama = _FakeOllama(local=tuple(setup_tags("qwen2.5-coder")))
    result = runner.invoke(app, ["setup", "--no-test"])
    assert result.exit_code == 0, result.output
    assert env.pulled == [] and env.asked == []
    assert "Already downloaded" in result.output


def test_setup_stops_clearly_when_ollama_is_not_running(env) -> None:
    env.ollama = _FakeOllama(available=False)
    result = runner.invoke(app, ["setup", "--yes"])
    assert result.exit_code == 1
    assert "https://ollama.com" in result.output
    assert env.pulled == []


def test_setup_offers_llama_cpp_and_installs_only_with_consent(env) -> None:
    env.llamacpp = False
    env.ollama = _FakeOllama(local=tuple(setup_tags("qwen2.5-coder")))
    result = runner.invoke(app, ["setup", "--no-test"], input="n\n")
    assert result.exit_code == 0, result.output
    assert env.installs == []
    result = runner.invoke(app, ["setup", "--no-test"], input="y\n")
    assert result.exit_code == 0, result.output
    assert env.installs and "llama-cpp-python" in env.installs[0]


def test_setup_model_override_and_small_machine(env) -> None:
    env.ram_gb = 6
    result = runner.invoke(app, ["setup", "--yes", "--no-test"])
    assert result.exit_code == 0, result.output
    assert _config(cli_main.CONFIG_PATH)["model"] == "qwen2.5:1.5b-instruct"
    result = runner.invoke(app, ["setup", "--yes", "--no-test", "--model", "llama3.2"])
    assert _config(cli_main.CONFIG_PATH)["model"] == "llama3.2"
    assert env.pulled[-1] == "llama3.2:3b-instruct-q4_K_M"
