# Architectures

All policy/value models return finite `policy_logits: [batch, 4672]` and
`value: [batch]`. The value is bounded to `[-1, 1]` from the side-to-move
perspective.
Legal masking remains external and mandatory.

## Implemented: Single-Board ResNet

`PolicyValueResNet` in
[src/mcchess/model/network.py](../src/mcchess/model/network.py) uses a convolutional
stem, residual tower, and separate policy/value heads. Input is
`[batch, 18, 8, 8]`.

| Preset | Channels | Residual blocks | Value hidden size | Normalization |
|---|---:|---:|---:|---|
| `resnet_a` | 64 | 4 | 128 | None |
| `resnet_b` | 64 | 6 | 128 | None |
| `resnet_c` | 128 | 10 | 256 | BatchNorm |

These are current built-in presets; historical run names do not override the
model config saved in each checkpoint. In particular, earlier runs named
ResNet-A used smaller inline configs.

The published [epoch-30 artifact](../models_archive/README.md) is ResNet-C with
**3,600,612 parameters**. Its exact config, checksum, source lineage, metrics,
and limitations are in [manifest.json](../models_archive/manifest.json).
It is inference-only.

A/B retain their no-normalization design. BatchNorm or other normalization
changes require explicit matched ablations. Completed training metrics alone
are not a complete architecture comparison.

## Deferred Architecture Studies

None of the following families is implemented. Tensor and metadata details
must be settled and documented before their milestones begin.

| Family | Proposed computation | Question |
|---|---|---|
| History ResNet | Stack history planes as `[batch, history_planes, 8, 8]`; retain current-position metadata. | Does raw board history help? |
| Square attention | ResNet features -> 64 square tokens with square embeddings -> Transformer encoder -> heads. | Does global square interaction help? |
| LSTM history | Shared CNN per board -> oldest-to-newest LSTM -> final state -> heads. | Does recurrent memory help? |
| LSTM with temporal attention | Current/newest LSTM output queries all LSTM outputs; attention, residual, LayerNorm, then heads. | Does selective history recall help? |
| Temporal Transformer | Shared CNN embeddings, time embeddings, Transformer encoder, and current-position token -> heads. | Does temporal attention help compared with recurrence? |

Sequence models would accept `[batch, history_length, board_planes, 8, 8]`.
History padding and value perspective stay anchored to the current position.
Check invalid history lengths and that attention-head count divides the hidden
dimension. [ROADMAP.md](../ROADMAP.md) records milestone gates.

## Later Optional: Sparse Accumulator

An NNUE-style study would use a McChess-defined sparse feature schema, not
imported Stockfish weights or external engine targets. Features must represent
side to move and document castling/en-passant information needed for policy
prediction. Any incremental accumulator must match full recomputation in tests.

Keep the policy/value output contract above. A value-only scorer would require
a separately documented experimental bot interface.

## Comparison Gate

Each new architecture needs a config, shape and invalid-input tests, finite
outputs, a backward smoke test, parameter count, and inference speed.
Comparisons must match dataset, split, seed, training budget, openings, and
search settings; report speed alongside accuracy or arena score. Use the
[evaluation protocol](EVALUATION_PROTOCOL.md).
