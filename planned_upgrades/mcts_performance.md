# MCTS Performance Plan

Status: in progress. Legal indexing once per expansion, legal-only softmax,
`torch.inference_mode()`, batched host transfers, and stored node visit totals
were implemented on 2026-07-15. The local before/after position check lacked
the reproducible config and counters below; it is not a reportable speed result.

Batched leaf inference also exists. Reserved paths can reach the same pending
node and share one neural evaluation. Simulation throughput alone can therefore
overstate distinct search work.

## Next: Measure Search Work

Add a config-driven benchmark over fixed opening, middlegame, and endgame
positions. Record checkpoint, device/hardware, config, seed, and Git commit,
plus:

- requested/completed simulations and distinct neural evaluations;
- terminal leaves, pending-leaf collisions, model calls, and actual batch sizes;
- selection, encoding, terminal-check, inference, expansion, and backup time.

Use batch size 1 as the behavioral reference. Compare fixed-simulation and
fixed-wall-clock results separately on the same checkpoint and hardware.

## Remaining Work, In Order

1. Resolve the direct-caller eval-mode contract. Checkpoint loading calls
   `model.eval()`, but `MCTSSearch` accepts an arbitrary model and does not set
   eval mode. Document or enforce that requirement and test BatchNorm/dropout
   behavior; `inference_mode()` alone does not disable training behavior.
2. Profile board copies and draw checks. Consider a push/pop working board
   while retaining only inputs and expansion data needed after inference.
   Preserve repetition history and all terminal/draw rules.
3. Measure collisions before changing reservations. Compare virtual loss with
   flushing a partial batch when no distinct leaf is available. Record the
   exact semantics; batch size 1 must retain deterministic behavior.
4. Add opt-in subtree reuse with an explicit game lifecycle. Reset state
   between games and report carried visits as part of the effective budget.
5. Consider batching independent games through one inference queue. Only then
   measure mixed precision, compilation, fixed buffers, or CUDA graphs.

## Acceptance

Each change needs same-hardware before/after timing and config-backed arena
comparisons. Saved artifacts must identify batching and tree-reuse settings.
Preserve legal-only expansion, normalized priors, value perspective and backup
signs, and tests for castling, promotions, en passant, checkmate, stalemate,
repetition, and fifty-move handling. Report distinct evaluations alongside
simulations and elapsed time; speed alone does not establish playing strength.
