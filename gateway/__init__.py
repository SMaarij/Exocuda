"""
gateway/ -- Track C: Platform & Revenue (Master Doc FR-07, Component 7).

Owns:
    - backend.py       -- CascadeCloudBackend, implementing shared.types.ExecutionBackend
    - providers/        -- wholesale GPU provider adapter (FR-07.4)
    - billing/           -- Stripe metering + spend caps (FR-07.2, FR-07.5)
    - api/                -- FastAPI gateway app

Guiding rule (Section 10, master doc): nothing in here imports from
engine/execution/ or engine/intelligence/ directly. gateway/backend.py
only imports the ExecutionBackend protocol from shared/types.py.
"""
