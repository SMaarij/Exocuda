"""
cli/tests/test_main.py -- Track C, Tasks 1, 12-14 (FR-08.1-08.3).

Every command is exercised via Typer's CliRunner with the real
HardwareProfiler/Pipeline construction monkeypatched out, so these
tests need no live Ollama, GPU, or network (NFR-09).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

import cli.main as cli_main
from cli.main import app
from shared.types import BackendResult, Message

runner = CliRunner()


class _FakePipeline:
    def __init__(self, text: str = "a fake answer", tier: str = "local") -> None:
        self._text = text
        self._tier = tier

    def run(self, messages: list[Message]) -> dict:
        return {
            "text": self._text, "tier": self._tier, "cache_hit": False, "confidence": 0.9,
            "escalated": False, "speculative_active": False, "tok_s": 20.0,
            "latency_ms": 42.0, "est_cost_usd": 0.0,
        }

    def run_stream(self, messages: list[Message]) -> Iterator[str]:
        yield self._text


class _FakeHardwareProfiler:
    def __init__(self, runner=None) -> None:
        pass

    def profile(self) -> dict:
        return {
            "schema_version": 1, "profiled_at": "2026-01-01T00:00:00Z", "os": "linux",
            "cpu_cores": 4, "ram_gb": 16.0, "gpu_vendor": None, "vram_gb": 0.0,
            "disk_throughput_mb_s": 500.0, "disk_free_gb": 100.0,
            "measured_tok_s": {"q4_K_M": 10.0}, "peak_memory_gb": {"q4_K_M": 2.0},
            "local_capable": True,
        }


class _FakeIncapableHardwareProfiler(_FakeHardwareProfiler):
    def profile(self) -> dict:
        result = super().profile()
        result["local_capable"] = False
        result["measured_tok_s"] = {"q4_K_M": 0.5}
        return result


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "CONFIG_PATH", tmp_path / "config.json")


@pytest.fixture(autouse=True)
def _fake_hardware_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.execution.hardware_profiler as hp_module
    import engine.execution.model_runner as runner_module

    monkeypatch.setattr(hp_module, "HardwareProfiler", _FakeHardwareProfiler)
    monkeypatch.setattr(runner_module, "OllamaReferenceModelRunner", lambda: None)


def test_init_writes_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "local_capable: True" in result.stdout
    assert cli_main.CONFIG_PATH.exists()


def test_init_warns_when_local_incapable(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.execution.hardware_profiler as hp_module

    monkeypatch.setattr(hp_module, "HardwareProfiler", _FakeIncapableHardwareProfiler)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "CascadeCloud" in result.stdout


def test_ask_prints_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "build_pipeline", lambda **kw: _FakePipeline(text="42"))
    result = runner.invoke(app, ["ask", "what is 6 times 7"])
    assert result.exit_code == 0
    assert "42" in result.stdout


def test_ask_no_log_flag_disables_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_build_pipeline(decision_logging: bool = True):
        captured["decision_logging"] = decision_logging
        return _FakePipeline()

    monkeypatch.setattr(cli_main, "build_pipeline", fake_build_pipeline)
    runner.invoke(app, ["ask", "hi", "--no-log"])
    assert captured["decision_logging"] is False


def test_chat_exits_cleanly_on_exit_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "build_pipeline", lambda **kw: _FakePipeline())
    result = runner.invoke(app, ["chat"], input="exit\n")
    assert result.exit_code == 0


def test_chat_echoes_pipeline_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "build_pipeline", lambda **kw: _FakePipeline(text="hello back"))
    result = runner.invoke(app, ["chat"], input="hi there\nexit\n")
    assert "hello back" in result.stdout


def test_stats_reports_no_requests_when_log_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", "--log-path", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 0
    assert "No requests logged yet" in result.stdout


def test_stats_aggregates_real_log_file(tmp_path: Path) -> None:
    from engine.intelligence.decision_log import log_decision

    log_path = tmp_path / "decisions.jsonl"
    log_decision(
        {"text": "a", "tier": "local", "cache_hit": True, "confidence": 0.9, "escalated": False,
         "speculative_active": False, "tok_s": 10.0, "latency_ms": 5.0, "est_cost_usd": 0.0},
        log_path,
    )
    result = runner.invoke(app, ["stats", "--log-path", str(log_path)])
    assert result.exit_code == 0
    assert "Cache hit rate:        100.0%" in result.stdout


def test_profile_prints_json_hardware_profile() -> None:
    result = runner.invoke(app, ["profile"])
    assert result.exit_code == 0
    assert "local_capable" in result.stdout


def test_load_config_returns_empty_dict_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "CONFIG_PATH", tmp_path / "nope.json")
    assert cli_main.load_config() == {}


def test_save_and_load_config_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "CONFIG_PATH", tmp_path / "config.json")
    cli_main.save_config({"key": "value"})
    assert cli_main.load_config() == {"key": "value"}
