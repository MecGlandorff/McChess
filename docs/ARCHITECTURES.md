# Architectures

This project compares architectures as controlled ablations.

All architectures must output:

```text
policy_logits: [batch, 4672]
value: [batch]
```

The value must be in `[-1, 1]`.

## 1. ResNet

Purpose:

- baseline policy/value model
- fast inference
- strong enough for MCTS experiments

Current implementation:

- `src/mcchess/model/network.py`
- class: `PolicyValueResNet`
- config: `ResNetConfig`
- presets: `resnet_a`, `resnet_b`

Input:

```text
[batch, 18, 8, 8]
```

Architecture:

```text
conv stem
residual tower
policy head
value head
```

No normalization layers are used in the current baseline. BatchNorm, GroupNorm,
or other normalization variants should be treated as matched ablations rather
than silently replacing the baseline architecture.

Outputs:

```text
policy_logits: [batch, 4672]
value: [batch]
```

Packaged presets:

- `resnet_a`: the default `ResNetConfig()` single-board model.
- `resnet_b`: a deeper compact single-board baseline with `channels=64`,
  `num_blocks=6`, and `value_hidden_dim=128`.
- `resnet_c` (planned): adds BatchNorm and related training refinements as a
  measured ablation on top of the deliberately minimal baselines.

`resnet_b` is a model package for controlled follow-up training and ablation. It
has no reported result until trained and evaluated under the project protocol.

## 2. History ResNet

Purpose:

- test whether raw board history helps

Input:

```text
[batch, history_planes, 8, 8]
```

where:

```text
history_planes = history_length * board_planes
```

## 3. ResNet + Square Attention

Purpose:

- test whether global square-to-square communication helps

Architecture:

```text
conv stem
residual tower
flatten 8x8 into 64 square tokens
square embeddings
TransformerEncoder over square tokens
policy/value heads
```

## 4. LSTM History

Purpose:

- test whether recurrent temporal memory helps

Input:

```text
[batch, history_length, board_planes, 8, 8]
```

Architecture:

```text
shared CNN board encoder
LSTM over board embeddings
final hidden state
policy/value heads
```

Internal sequence order should be oldest-to-newest.

## 5. LSTM + Temporal Attention

Purpose:

- test whether selective recall over recurrent states helps

Architecture:

```text
shared CNN board encoder
LSTM over board embeddings
current-query MultiheadAttention over LSTM outputs
LayerNorm + residual
policy/value heads
```

The current/newest LSTM output is the query.

All LSTM outputs are keys and values.

## 6. Temporal Transformer

Purpose:

- test whether full temporal attention beats recurrence

Architecture:

```text
shared CNN board encoder
time embeddings
TransformerEncoder over board embeddings
current-position token
policy/value heads
```

## Later Optional: NNUE-Style Sparse Accumulator

Purpose:

- test whether sparse chess features plus a compact accumulator can improve
  inference speed or strength per second under limited compute
- compare a non-convolutional inductive bias against the ResNet and temporal
  families

This must be a McChess-defined architecture. Do not import Stockfish NNUE
weights, train from engine evaluations, or use external best-move labels.

Possible input:

```text
[batch, num_sparse_features]
```

or an incremental accumulator state derived from a documented sparse feature
schema.

Required outputs remain:

```text
policy_logits: [batch, 4672]
value: [batch]
```

Design constraints:

- legal move masking remains external and mandatory
- feature encoding must include side-to-move perspective explicitly
- castling and en-passant information must be represented if needed for policy
  prediction
- incremental accumulator updates, if implemented, must match full
  recomputation in tests
- report parameter count and positions/sec alongside loss or arena results

An NNUE-style value-only scorer can be studied as a separate experimental bot,
but it must be documented as value-only and cannot silently replace the
policy/value model interface used by neural policy and MCTS components.

## Architecture Comparison Requirements

Each architecture needs:

- config example
- parameter count logging
- forward shape test
- backward smoke test
- finite output test
- inference speed measurement if possible
