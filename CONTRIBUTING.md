# Contributing to Cascade

Thanks for helping. Bug reports, benchmark results from your hardware, docs fixes and code are all
welcome.

## Set up

```bash
git clone https://github.com/SMaarij/Exocuda.git
cd Exocuda
pip install -e ".[dev]"
pytest
```

The whole suite runs in seconds and needs no GPU, no running Ollama and no network. Keep it that
way: anything that talks to a model, a server or the internet is injected through a protocol and
replaced by a fake in tests (see `engine/intelligence/tests/fakes.py`, `gateway/tests/fakes.py`).

## Ground rules

- **Report only what was measured.** Speed, acceptance rates and accuracy figures must come from a
  real run. No hard-coded speedup factors, no estimates presented as results. If a number is from
  a small fixture, say so.
- **Nothing leaves the machine unless the user configured it.** No default remote endpoints, no
  telemetry, and no network call on the local path except pulling a model that's missing.
- **Secrets stay in environment variables.** Never write an API key to a config file, a log or the
  decision log.
- **The seams are stable.** `shared/types.py` (`ExecutionBackend`, `BackendResult`,
  `PipelineResult`, `HardwareProfile`) is what the components agree on. If you change it, add an
  entry to `cascade_local/shared/CHANGELOG.md` explaining why.
- **Tests with every change.** New behaviour gets a test; a fixed bug gets a test that would have
  caught it.

## Sharing benchmark results

The most useful contribution right now is numbers from hardware other than one CPU-only laptop,
especially machines with a GPU:

```bash
cascade init                                    # hardware profile and local tok/s
python scripts/benchmark_acceleration.py        # speculative decoding, needs the llamacpp extra
```

Open an issue with the output, your OS, CPU/GPU, RAM and the llama-cpp-python build you used
(CPU, CUDA, Metal...). Hardware ladder results go in `docs/hardware_compat_results.md`.

## Pull requests

1. Branch from `main`.
2. Keep the change focused; explain the why in the description.
3. Make sure `pytest` passes. CI runs it on Linux, macOS and Windows.

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).
