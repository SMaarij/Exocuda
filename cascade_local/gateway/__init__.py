"""
gateway/ -- everything that talks to an OpenAI-compatible endpoint.

    byok.py          -- CloudBYOKBackend: the `cloud_byok` tier (your own provider key)
    backend.py       -- SelfHostedBackend: the `self_hosted` tier's client
    api/, store.py,  -- the self-hosted Cascade server: run it on a GPU box and
    serve.py            share it with your team (API keys, optional token quotas)
    providers/       -- the HTTP client both tiers and the server use

Nothing in here imports from engine/; backends only implement the
ExecutionBackend protocol from shared/types.py.
"""
