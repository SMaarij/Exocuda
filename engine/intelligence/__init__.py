"""
engine/intelligence/

Track B -- Intelligence Layer (Master Doc Section 10.1: FR-04, FR-05, FR-06).

Owns:
    - Pipeline            (pipeline.py)          -- the Section 9 orchestrator seam
    - ContextCompressor    (context_compressor.py) -- FR-04, not yet built
    - SemanticRouter        (router/)              -- FR-05, not yet built
    - SemanticCache          (cache.py)              -- FR-06, not yet built

Guiding rule (Section 10, master doc): nothing in here imports from
gateway/ or cli/. The one accepted, documented exception is
LocalExecutionBackendError from engine/execution/backend.py -- see the
top-of-file note in pipeline.py for why, and shared/CHANGELOG.md for
the cleanup this flags for a future revision.
"""
