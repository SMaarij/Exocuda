# Security

## Reporting a vulnerability

Please don't open a public issue for a security problem. Use GitHub's private
[vulnerability reporting](https://github.com/SMaarij/Exocuda/security/advisories/new) instead, with
steps to reproduce and the impact you see. We'll acknowledge it as soon as we can and credit you
in the fix unless you'd rather not be named.

## Scope notes

- **Self-hosted server (`gateway/`).** API keys are stored as SHA-256 hashes only. The server
  speaks plain HTTP; run it behind HTTPS before exposing it beyond a trusted network. It holds the
  upstream's API key — protect the machine and its environment accordingly.
- **Local data.** `~/.cascade/` holds the config, the decision log (routing metadata only) and the
  cache, which stores your prompts and answers in plain text. Treat it like any other private file.
- **Keys.** Cascade reads provider keys from environment variables and refuses to write them to its
  config file.
