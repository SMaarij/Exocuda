# shared/ CHANGELOG

Durable record of any change to either seam (`Pipeline` or
`ExecutionBackend`) or to `HardwareProfile`/`BackendResult`/`PipelineResult`.
Every entry below must have been reviewed by all three tracks before merge
(Section 10.3). This file is the source of truth over anything said on a
call or in chat.

## [Unreleased] -- Phase 1, Week 1

### Added (Track A, proposed for J-1 freeze)

- `shared/types.py` created with `Message`, `HardwareProfile`,
  `BackendResult`, `PipelineResult`, and the `ExecutionBackend` Protocol.
- `HardwareProfile` extends the Section 9 sketch with the full field set
  FR-01.1 requires (`cpu_cores`, `gpu_vendor`, `disk_throughput_mb_s`,
  `disk_free_gb`, `peak_memory_gb`, `schema_version`, `profiled_at`, `os`)
  in addition to the `ram_gb` / `vram_gb` / `measured_tok_s` /
  `local_capable` fields shown in the master doc's illustrative snippet.
  Rationale: FR-01.1 explicitly requires the system to detect all of
  these; Section 9's snippet was a minimal illustration of the seam
  shape, not an exhaustive schema. Added `schema_version` (starts at
  `HARDWARE_PROFILE_SCHEMA_VERSION = 1`) so ModelLadder and future
  consumers can detect an incompatible profile instead of guessing at
  missing fields.
- `ExecutionBackend.generate()` / `.generate_stream()` signatures match
  the master doc exactly (list[Message] -> BackendResult /
  Iterator[str]).

### Status

- **Pending J-1 sign-off** -- per Section 13.1, all three tracks must
  agree on and freeze this file before further engine/, gateway/, or
  platform code is written. This version is Track A's proposal, built
  against the FR-01/FR-03 requirements it owns; Track B and Track C
  review needed before it is marked frozen.
- Once frozen, any further change to this file requires a new dated
  entry here plus all three tracks' review on the PR (Section 10.3),
  no exceptions.
