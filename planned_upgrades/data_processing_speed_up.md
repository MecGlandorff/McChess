# Data Processing Speed-Up Plan

## Context

The current PGN-to-JSONL pipeline is correctness-first and intentionally simple.
That was appropriate for the initial dataset-builder milestone, but it is too
slow for repeated full-month Lichess processing.

Observed local signal:

- A 1,000-game January 2013 sample took roughly 15 minutes to process.
- The raw 1,000-game PGN was about 0.78 MB.
- The processed JSONL shards were about 11.9 MB total.
- The full January 2013 PGN was about 92.8 MB.
- The full-month processed train shard grew to hundreds of MB while processing.

The practical target for future work is at least 100x faster end-to-end dataset
preparation, without changing research scope or using engine labels.

## Constraints

Must preserve:

- human PGN moves/results only
- no Stockfish, Syzygy, tablebase, Leela, or external engine labels
- `python-chess` as the source of legal chess rules
- game-level splitting, never position-level splitting
- deterministic reproducibility from config and seed
- manifest provenance, filters, counts, split metadata, and code version
- value targets from side-to-move perspective
- legal move-index contract and tests

Any faster format must either remain auditable itself or be traceable back to an
auditable source manifest.

## Current Hot Path

Current processing flow:

```text
PGN file
  -> chess.pgn.read_game(...)
  -> for each mainline move:
       board.fen()
       move.uci()
       move_to_index(board, move)
       value target
       json.dumps(sample)
       shard.write(line)
  -> reread source PGN for checksum
  -> manifest
```

Important current code paths:

- `src/mcchess/data/pgn_reader.py`
  - `chess.pgn.read_game(stream)`
  - per-ply FEN generation
  - per-ply `move_to_index(board, move)`
- `src/mcchess/board/move_index.py`
  - `move_to_index()` rechecks `move in board.legal_moves`
- `src/mcchess/data/dataset_builder.py`
  - per-sample `json.dumps(...)`
  - per-sample line writes
  - byte-level progress wrapper
  - final `source.read_bytes()` checksum
- `src/mcchess/data/torch_dataset.py`
  - loads all JSONL rows into memory
  - parses FEN back into `chess.Board` on every sample access
  - re-encodes boards every epoch

## Likely Bottlenecks

Expected primary costs:

- pure-Python PGN parsing in `python-chess`
- repeated legal move generation during `move_to_index()`
- FEN string generation for every ply
- JSON serialization for every ply
- large write amplification from PGN to JSONL
- Windows filesystem and antivirus scanning of large JSONL writes

The progress bar can add overhead because it wraps `readline()` and encodes
each line to count bytes, but this is probably secondary.

## Stage 0: Measure Before Changing

Goal: get reliable baseline numbers.

Actions:

- Add or run a profiling command over the 1,000-game sample.
- Capture wall time, positions/sec, games/sec, output MB/sec, and CPU usage.
- Use `cProfile` first, then optionally `py-spy` if available.
- Record raw PGN size, output shard sizes, number of games, and number of
  positions.

Suggested metrics:

```text
raw_size_bytes
output_size_bytes
games_read
games_used
positions_emitted
elapsed_seconds
games_per_second
positions_per_second
raw_mb_per_second
output_mb_per_second
```

Acceptance:

- A baseline report exists for the 1,000-game sample.
- The top cumulative-time functions are identified.
- No behavior changes are made in this stage.

## Stage 1: Low-Risk Single-Process Improvements

Goal: improve throughput without changing dataset semantics or introducing
parallelism.

Candidate changes:

- Replace byte-level progress with game/position progress.
- Update progress every N games or N positions, not every input line.
- Buffer JSONL writes with per-game or chunked `writelines()`.
- Use compact JSON separators: `json.dumps(sample, separators=(",", ":"))`.
- Compute source SHA256 during the main source read if practical.
- Avoid `source.read_bytes()` for large files; stream checksum in chunks.
- Add an internal trusted move-index helper that skips repeated
  `move in board.legal_moves` only when called from validated PGN parsing.

Notes on trusted move indexing:

- Public `move_to_index(board, move)` should keep legality validation.
- A private helper can encode already-validated moves for dataset building.
- Tests must prove the trusted helper matches public `move_to_index()` on legal
  moves across ordinary moves, castling, promotion, and en passant.

Expected improvement:

- Possibly 2x-10x depending on where profiling points.
- Not expected to reach 100x alone.

Acceptance:

- Existing dataset-builder tests pass.
- New tests cover trusted move-index equivalence if added.
- Dataset manifest counts match existing behavior on fixtures.
- Sample output remains schema-compatible.
- Processing speed improves on the same 1,000-game sample.

## Stage 2: Faster Auditable Intermediate Format

Goal: reduce JSON/FEN overhead while keeping an auditable source of truth.

Options:

- Keep JSONL/FEN as the canonical small/debug format.
- Add an optional compact training cache for large runs.
- Store per-position arrays in a binary format such as `.npz` shards.
- Store model-ready board tensors only if the manifest records encoder version
  and tensor contract.
- Alternatively store compact board state fields plus policy/value targets.

Important tradeoff:

- Current docs say FEN remains the serialized dataset source of truth.
- If a binary training cache is added, update `DESIGN.md`,
  `INVARIANTS.md`, and `docs/DATASET_PROTOCOL.md`.
- Keep a manifest link from cache shards back to the PGN/JSONL source.

Expected improvement:

- Lower disk usage.
- Faster training startup and epochs.
- Faster repeated experiments after the first build.

Acceptance:

- JSONL/FEN path still works.
- Binary/cache path has shape and dtype tests.
- Manifest records cache schema version and encoder version.
- Training can load either JSONL or cache from config.

## Stage 3: Multiprocess PGN Processing

Goal: reach large speedups by using multiple CPU cores.

Proposed design:

```text
raw PGN
  -> split by game boundaries into chunk files or byte ranges
  -> worker processes parse chunks independently
  -> each worker writes train/val/test fragment files
  -> coordinator merges fragments
  -> coordinator writes combined manifest
```

Key requirements:

- Split by game, not by position.
- Do not split inside a PGN game.
- Do not rely on sequential RNG state shared across workers.
- Assign split deterministically from stable game identity.
- Keep counters per worker and merge them.
- Record worker count and chunking strategy in manifest.

Recommended split assignment:

```text
split_key = hash(seed, source_id, game_index_or_stable_game_id)
```

Then map the normalized hash to train/val/test ratios.

Why:

- Sequential RNG is awkward in parallel processing.
- Hash-based splitting is deterministic independent of worker count.
- Worker count can change without changing splits if game IDs are stable.

Open design question:

- Use sequential `game_index` assigned by pre-scan, or derive a stable ID from
  PGN headers/move text?

Recommendation:

- For first parallel implementation, pre-scan game byte offsets and assign
  global game indexes.
- Later consider normalized move-sequence hashes for duplicate handling.

Expected improvement:

- Roughly proportional to CPU cores after single-process overhead is reduced.
- With 8-16 cores plus Stage 1 improvements, 50x-100x becomes plausible.

Acceptance:

- Parallel and single-process builds produce equivalent sample counts on a
  fixture PGN.
- Splits are deterministic across worker counts.
- No game appears in more than one split.
- Corrupt/unknown games are counted correctly.
- Manifest records chunk count, worker count, split method, and counters.

## Stage 4: Streaming Training Loader

Goal: avoid loading full JSONL shards into memory and avoid reparsing FEN every
epoch when possible.

Candidate changes:

- Add iterable dataset support for large JSONL shards.
- Add memory-mapped or chunked binary cache loader.
- Use `num_workers > 0` safely in training configs.
- Consider pre-encoded tensor cache for repeated training runs.

Current issue:

- `SupervisedChessDataset` reads all rows into memory.
- `__getitem__` reconstructs `chess.Board(fen)` and encodes on every access.

Expected improvement:

- Faster startup for large datasets.
- Better epoch throughput.
- Lower memory pressure.

Acceptance:

- Existing JSONL dataset behavior remains available.
- New loader has tests for shape, dtype, policy index, and value dtype.
- Training script supports the loader through config.
- Repeated epoch throughput improves on the same dataset.

## Stage 5: Operational Improvements

Goal: make large runs easier to monitor and resume.

Candidate changes:

- Write progress by games and positions.
- Periodically write a temporary status JSON.
- Write output fragments then atomically finalize shard paths.
- Detect existing complete outputs and require `--force` to overwrite.
- Support resume only after a clear design, not ad hoc appends.
- Warn if output split files remain empty after enough games.

Acceptance:

- Interrupted runs are clearly marked incomplete.
- Completed runs have final manifest and non-empty expected shards.
- Re-running without `--force` does not silently overwrite completed datasets.

## Test Plan

Tests to add or preserve:

- tiny valid PGN still emits expected samples
- unknown result handling
- corrupt game skipping
- split by game, never by position
- deterministic splits across worker counts
- side-to-move value perspective
- policy target index generation
- move-index equivalence for trusted/internal path
- castling, promotion, en passant in optimized path
- manifest count consistency
- output fragment merge consistency
- cache loader shape and dtype if binary cache is added

## Benchmark Plan

Use three scales:

- fixture: tiny unit tests
- sample: first 1,000 January 2013 games
- full small month: full January 2013 archive

Report:

```text
config
git commit
python version
platform
CPU
storage notes
raw PGN size
games read/used/skipped
positions emitted
output size
elapsed seconds
games/sec
positions/sec
speedup versus baseline
```

Do not compare playing strength or Elo from this work. This is only a data
pipeline performance upgrade.

## Suggested Implementation Order

1. Add profiling script or documented profiling command.
2. Replace byte-progress wrapper with game/position progress.
3. Buffer writes and stream checksum.
4. Add trusted internal move-index helper with tests.
5. Benchmark Stage 1.
6. Design hash-based split assignment.
7. Implement PGN chunk pre-scan.
8. Implement multiprocess worker fragments.
9. Merge fragments and manifests.
10. Benchmark Stage 3.
11. Add optional training cache/loader only after the canonical JSONL path is
    stable and measured.

## Non-Goals

- No engine labels.
- No tablebase labels.
- No position-level split shortcuts.
- No unsupported strength claims.
- No large framework adoption unless profiling proves the need.
- No silent replacement of the auditable dataset format.
