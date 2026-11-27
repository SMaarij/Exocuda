# Changelog

User-facing changes. Measurements behind every number are in
[docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md), and the final summary is
[docs/FINDINGS.md](docs/FINDINGS.md).

## 0.12.0 — 2026-11-27

First release meant for other people to install.

- **Install with `pip install cascade-local`.** The PyPI name is `cascade-local`, because
  `cascade` is taken; the command is still `cascade`. Everything now lives in one Python package,
  `cascade_local`, so installing Cascade no longer adds generic top-level modules (`cli`,
  `engine`, `gateway`, `shared`) that could clash with other packages.
- **`cascade setup`** sets everything up in one command:
  - checks Ollama;
  - picks models for your RAM: qwen2.5-coder 7B with 12 GB or more, qwen2.5 3B with 8 GB,
    qwen2.5 1.5B below that, each with the qwen2.5 0.5B helper;
  - downloads them, asking first;
  - offers to install llama-cpp-python, which turns on speculative decoding;
  - measures the machine and asks a test question.
- **`cascade stats`** now shows who answered (cache, small model, main model, cloud), typical
  answer times, and the time the cache saved you. Add `--days N` for a recent window.
- **`cascade cache list / forget / clear`** lets you see cached answers and remove a wrong one
  (`cascade cache forget --last`, by ID, or `--match TEXT`), or empty the cache. A running Cascade
  server picks up the change immediately.
- Plain-language help for every command.

**Upgrading from a source checkout:** run `pip uninstall cascade`, then `pip install -e .`. If
you used `cascade install`, run it again so the login entry uses the new module path.

## 0.11.0 and earlier

Development releases, from a source checkout only:
- lossless speculative decoding with cost-aware draft length (1.5–1.6x, about 1.9x on code);
- the semantic cache with the 7B judging close matches;
- small/big model routing;
- the Ollama and OpenAI drop-in server and `cascade install --as-ollama`;
- streaming;
- saved conversation state;
- the 7B's resident RAM cut from 7.2 to 4.6 GB.

See [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) Addenda 1–28.
