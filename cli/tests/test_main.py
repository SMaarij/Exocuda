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
    def __init__(self, runner=None, **kwargs) -> None:
        pass

    def load_cached_profile(self):
        return None

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
    assert "configure a cloud tier" in result.stdout


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


# ---- config / tier wiring / doctor -------------------------------------------------


def test_config_set_show_unset_round_trip() -> None:
    assert runner.invoke(app, ["config", "set", "byok.model", "gpt-4o-mini"]).exit_code == 0
    assert runner.invoke(app, ["config", "set", "byok.input_price_per_1k", "0.15"]).exit_code == 0
    assert cli_main.load_config()["byok"] == {"model": "gpt-4o-mini", "input_price_per_1k": 0.15}
    assert '"gpt-4o-mini"' in runner.invoke(app, ["config", "show"]).output
    runner.invoke(app, ["config", "unset", "byok.model"])
    assert "model" not in cli_main.load_config()["byok"]


def test_config_refuses_to_store_a_secret() -> None:
    result = runner.invoke(app, ["config", "set", "byok.api_key", "sk-live-123"])
    assert result.exit_code == 1
    assert "sk-live-123" not in cli_main.CONFIG_PATH.read_text() if cli_main.CONFIG_PATH.exists() else True


def test_byok_settings_resolve_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"byok": {"base_url": "https://api.openai.com", "model": "gpt-4o-mini", "api_key_env": "MY_KEY"}}
    monkeypatch.setenv("MY_KEY", "sk-test")
    assert cli_main._byok_settings(config)["api_key"] == "sk-test"
    assert cli_main._byok_settings({}) is None


def test_self_hosted_is_not_configured_by_default() -> None:
    """No default endpoint: prompts never go to a server the user didn't choose (NFR-05)."""
    assert cli_main._self_hosted_settings({}) is None
    assert cli_main._self_hosted_settings({"self_hosted": {"url": "http://localhost:8000"}})["base_url"] == "http://localhost:8000"


def test_build_local_backend_honours_runner_and_acceleration(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.execution.llamacpp_runner as lr

    monkeypatch.setattr(lr, "LlamaCppRunner", lambda draft_model_tag, **kwargs: _AcceleratedStub())
    backend = cli_main.build_local_backend({"runner": "llamacpp", "acceleration": "draft_model"})
    from engine.execution.acceleration_engine import AccelerationLayer

    assert backend._engine._preferred_layer == AccelerationLayer.DRAFT_MODEL
    assert cli_main.build_local_backend({})._model_id == "llama3.2"


class _AcceleratedStub:
    def load(self, m, q): ...
    def generate(self, p, n): ...
    def unload(self): ...
    def set_acceleration(self, mode): return True


def test_footer_shows_tier_speed_and_cost() -> None:
    footer = cli_main._status_footer({"tier": "local", "cache_hit": False, "tok_s": 12.34, "latency_ms": 900.0, "est_cost_usd": 0.0, "escalated": False})
    assert footer == "[local | 12.3 tok/s | 900ms | $0.0000]"


def test_ask_stream_prints_streamed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "build_pipeline", lambda **kw: _FakePipeline(text="streamed answer"))
    assert "streamed answer" in runner.invoke(app, ["ask", "hi", "--stream"]).output


def test_doctor_reports_missing_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.execution.model_runner as mr

    class _DownOllama:
        def is_available(self): return False

    monkeypatch.setattr(mr, "OllamaReferenceModelRunner", lambda: _DownOllama())
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "Ollama server reachable" in result.output


# ---- Settings added after the 2026-11-06 live benchmark ---------------------


@pytest.fixture
def captured_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict:
    """build_pipeline() with every heavy part faked; returns the kwargs it passed."""
    import engine.intelligence.cache as cache_module
    import engine.intelligence.embeddings as emb_module
    import engine.intelligence.pipeline as pipeline_module
    from engine.intelligence.tests.fakes import FakeEmbedder

    captured: dict = {}
    monkeypatch.setattr(emb_module, "FastEmbedEmbedder", FakeEmbedder)
    monkeypatch.setattr(cache_module, "SemanticCache", lambda embedder, config=None: ("cache", config))
    monkeypatch.setattr(pipeline_module, "Pipeline", lambda **kw: captured.update(kw) or "pipeline")
    monkeypatch.setattr(cli_main, "build_local_backend", lambda config: "local")
    return captured


def test_default_system_prompt_is_sent(captured_pipeline: dict) -> None:
    cli_main.build_pipeline(decision_logging=False)
    assert captured_pipeline["system_prompt"] == cli_main.DEFAULT_SYSTEM_PROMPT
    assert "uncached_tiers" not in captured_pipeline  # Pipeline's own default applies


def test_system_prompt_can_be_replaced_or_disabled(captured_pipeline: dict) -> None:
    runner.invoke(app, ["config", "set", "system_prompt", "Talk like a pirate."])
    cli_main.build_pipeline(decision_logging=False)
    assert captured_pipeline["system_prompt"] == "Talk like a pirate."

    runner.invoke(app, ["config", "set", "system_prompt", '""'])
    cli_main.build_pipeline(decision_logging=False)
    assert captured_pipeline["system_prompt"] == ""


def test_cache_settings_come_from_config(captured_pipeline: dict) -> None:
    runner.invoke(app, ["config", "set", "cache.similarity_threshold", "0.95"])
    runner.invoke(app, ["config", "set", "cache.uncached_tiers", '["self_hosted", "cloud_byok"]'])
    cli_main.build_pipeline(decision_logging=False)
    _, cache_config = captured_pipeline["cache"]
    assert cache_config.similarity_threshold == 0.95
    assert captured_pipeline["uncached_tiers"] == ("self_hosted", "cloud_byok")


def test_local_max_tokens_comes_from_config() -> None:
    assert cli_main.build_local_backend({"local": {"max_tokens": 128}})._max_tokens == 128
    assert cli_main.build_local_backend({})._max_tokens == 512


def test_ask_warns_before_the_first_run_hardware_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "build_pipeline", lambda decision_logging=True: _FakePipeline())
    result = runner.invoke(app, ["ask", "hi"])
    assert "one-time hardware check" in result.output


def test_ask_is_quiet_once_profiled(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.execution.hardware_profiler as hp_module

    class _Profiled(_FakeHardwareProfiler):
        def load_cached_profile(self):
            return self.profile()

    monkeypatch.setattr(hp_module, "HardwareProfiler", _Profiled)
    monkeypatch.setattr(cli_main, "build_pipeline", lambda decision_logging=True: _FakePipeline())
    assert "hardware check" not in runner.invoke(app, ["ask", "hi"]).output


def test_every_cache_setting_is_configurable(captured_pipeline: dict) -> None:
    runner.invoke(app, ["config", "set", "cache.term_swap_guard", "lenient"])
    runner.invoke(app, ["config", "set", "cache.min_shared_prefix", "5"])
    runner.invoke(app, ["config", "set", "cache.time_sensitive_patterns", '["\\bstock\\b"]'])
    cli_main.build_pipeline(decision_logging=False)
    _, cache_config = captured_pipeline["cache"]
    assert cache_config.term_swap_guard == "lenient"
    assert cache_config.min_shared_prefix == 5
    assert cache_config.time_sensitive_patterns == ("\bstock\b",)


def test_draft_settings_default_to_adaptive() -> None:
    assert cli_main._draft_settings({}) == {"adaptive_draft": True}


def test_draft_settings_read_config() -> None:
    config = {"draft": {"adaptive": "false", "tokens": "8", "max_tokens": 12}}
    assert cli_main._draft_settings(config) == {"adaptive_draft": False, "draft_tokens": 8, "max_draft_tokens": 12}


def test_profile_command_measures_the_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.execution.hardware_profiler as hp_module

    seen: dict = {}

    class _Recording(_FakeHardwareProfiler):
        def __init__(self, runner=None, **kwargs) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(hp_module, "HardwareProfiler", _Recording)
    cli_main.save_config({"runner": "llamacpp", "model": "qwen2.5-coder"})

    assert runner.invoke(app, ["profile"]).exit_code == 0
    assert seen["reference_model_id"] == "qwen2.5-coder"
    assert seen["quant_levels"][0] == "7b-instruct-q4_K_M"
