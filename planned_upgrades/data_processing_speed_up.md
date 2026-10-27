# Data Processing Performance Plan

Status: profiling and PGN preparation improvements remain open. Optional
memory-mapped tensor caches already avoid FEN parsing and encoding during
repeated training; their format and restart behavior are in the
[dataset protocol](../docs/DATASET_PROTOCOL.md).

Earlier unarchived local observations were roughly 15 minutes for 1,000 January
2013 games, 0.78 MB of raw PGN, and 11.9 MB of JSONL output. The full-month PGN
was about 92.8 MB and its train shard grew to hundreds of MB. These observations
lack a reproducible timing protocol and are not a current speed baseline.

## Measure First

Profile the current builder on the same 1,000-game sample, then the full small
month. Record config, commit, Python/platform, CPU/storage, source checksum and
size, games read/used/skipped, positions, output size, elapsed time, games/sec,
and positions/sec. Use cumulative-time profiling to identify the bottleneck
before setting a speedup target.

Candidate costs are PGN parsing, repeated legal validation in move indexing,
per-ply FEN/JSON generation, writes, byte-progress accounting, and the final
whole-file checksum read.

## Incremental Changes

1. Benchmark throttled game/position progress, buffered compact JSONL writes,
   and streamed checksumming. Preserve the serialized schema and counters.
2. If profiling supports it, add a private encoder for already-validated moves.
   Public `move_to_index` must keep legality checks; test equivalence on ordinary
   moves, castling, promotions, and en passant.
3. Only after the single-process baseline is measured, design parallel PGN
   chunks at game boundaries. Pre-scan offsets/global game indices, assign
   splits from stable identity and seed, write worker fragments, then merge
   counters and shards. Test identical splits across worker counts. A new
   hash-based split rule changes the existing sequential-RNG split assignment:
   version and record it rather than silently replacing historical datasets.
4. Consider a streaming JSONL training loader if needed beyond the existing
   tensor-cache path. Keep FEN/JSONL auditable and caches traceable to source,
   schema, encoder shape, and dtype.
5. Design dataset-builder progress, atomic finalization, overwrite protection,
   and resume explicitly. Tensor-cache resume already exists; PGN preparation
   must not resume through unvalidated appends.

## Acceptance

Compare fixture counts and sample semantics before timing improvements. Test
corrupt/unknown-game handling, game-level splits, side-to-move targets, policy
indices, manifest counts, fragment merging, and any new loader shape/dtype.
Record worker count, chunking and split method if parallelism is added.

Human PGN targets, `python-chess` legality, deterministic reproducibility, and
manifest provenance remain mandatory. No engine/tablebase labels, position-level
splits, silent format changes, or unmeasured strength claims belong in this work.
