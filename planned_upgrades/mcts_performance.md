# MCTS Performance Plan

Status: in progress

Step 1 was implemented on 2026-07-15. A local fixed-position development check
was run before and after the change, but the reproducible benchmark config and
detailed counters described below are still pending. Do not treat the local
timings as a reportable performance result.

## Why This Work Is Needed

The current MCTS is a small, understandable baseline. It loads the model once
and evaluates one leaf per model call, but each search still does substantial
Python and board-management work around the model call.

Batching leaf inference is the first proposed optimization, but it needs careful
interpretation. Several reserved paths can reach the same unexpanded node. Those
paths may count toward the search budget even though the model evaluates that
position once. A faster reported simulation rate could therefore mean less
distinct neural search work.

The goal is to reduce move latency and increase arena throughput without
changing chess rules or hiding changes in search behavior.

## Measure First

Add a repeatable benchmark over fixed opening, middlegame, and endgame
positions. Record:

- requested and completed simulations
- distinct neural leaf evaluations
- terminal leaves and pending-leaf collisions
- model calls and actual inference batch sizes
- time spent in selection, board encoding, terminal checks, model inference,
  expansion, and backup
- checkpoint, device, hardware, config, seed, and git commit when available

Report fixed-simulation and fixed-wall-clock results separately. Batch size 1
is the behavioral reference for search changes.

## Planned Steps

### 1. Remove Avoidable Hot-Path Work (implemented)

- use `torch.inference_mode()` for model calls
- enumerate legal moves and their policy indices once per expansion
- apply softmax only to legal logits while keeping explicit legality masking
- transfer batch results from the device once instead of reading individual
  policy values with repeated `.item()` calls
- store node visit totals instead of recomputing them from every edge

This step must not require new checkpoints or retraining.

### 2. Reduce Board And Terminal Overhead

Use one working board with `push()` and `pop()` while collecting leaves. Store
the encoded input and legal expansion data needed after inference instead of a
full board copy for every pending path.

Terminal checks must retain checkmate, stalemate, repetition, fifty-move, and
insufficient-material behavior. Draw handling should be profiled and tested
before its implementation changes.

### 3. Define Batched Search Semantics

Track completed simulations and distinct neural evaluations as separate
metrics. Measure pending-leaf collisions, then compare proper virtual loss with
flushing a partially filled batch when no distinct leaf is available.

Batch size 1 must keep the current deterministic behavior. Larger batch sizes
may change search trajectories, so their arena results must record the exact
batching configuration.

### 4. Reuse Search Trees

Retain the reachable subtree after a played move and reroot it after the
opponent replies. Add an explicit game lifecycle so state cannot leak between
arena games. Tree reuse should start behind a config option because carried
visits change the effective search budget.

### 5. Batch Across Independent Searches

For arena evaluation and later data generation, allow independent games to
submit leaves to one model inference queue. This can fill GPU batches with
distinct positions without forcing a large speculative batch inside one tree.

Mixed precision, `torch.compile`, fixed buffers, and CUDA graphs should be
measured only after the Python search path is instrumented and simplified.

## Correctness Gates

- all expanded moves come from `python-chess` legal moves
- policy priors are normalized over legal moves only
- side-to-move value perspective and backup sign flips remain unchanged
- castling, promotion, en passant, checkmate, stalemate, repetition, and
  fifty-move behavior stay tested
- batch size 1 remains deterministic on fixed positions
- performance comparisons report distinct neural evaluations as well as
  simulations and wall-clock time

## Done Criteria

- the benchmark and counters are reproducible from config
- each optimization has before-and-after timing on the same checkpoint and
  hardware
- fixed-simulation and fixed-wall-clock arena comparisons are recorded
- batching and tree-reuse settings are present in saved evaluation artifacts
- no model architecture change or unsupported strength claim is mixed into the
  performance work
