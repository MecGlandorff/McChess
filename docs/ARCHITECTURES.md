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

Input:

```text
[batch, planes, 8, 8]
```

Architecture:

```text
conv stem
residual tower
policy head
value head
```

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

## Architecture Comparison Requirements

Each architecture needs:

- config example
- parameter count logging
- forward shape test
- backward smoke test
- finite output test
- inference speed measurement if possible
