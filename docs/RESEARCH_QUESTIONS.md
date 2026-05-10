# Research Questions

## Main Question

How far can a compact neural chess engine go using only human games, temporal modeling, neural search, and optional self-play without engine labels?

## Study 1 - Representation

Question:

Does temporal context improve supervised policy/value learning?

Compare:

- single-board encoding
- history-plane encoding
- sequence-of-boards encoding

Metrics:

- policy top-1/top-3/top-5
- value MSE
- value calibration
- training speed
- inference speed

## Study 2 - Architecture

Question:

Which architecture gives the best policy/value quality under limited compute?

Compare:

- ResNet
- History ResNet
- ResNet + square attention
- LSTM history
- LSTM + temporal attention
- Temporal Transformer

## Study 3 - Search

Question:

Which architecture benefits most from MCTS?

Compare each model at:

- policy only
- MCTS-25
- MCTS-50
- MCTS-100
- MCTS-400

Important metric:

- win rate per second, not only win rate per move

## Study 4 - Search Distillation

Question:

Can MCTS-generated targets improve a compact policy/value network without engine labels?

Compare:

- supervised human move target
- MCTS visit distribution target
- human + MCTS hybrid target

## Study 5 - Failure Analysis

Question:

Where do models fail?

Analyze:

- hanging pieces
- missed mate in one
- unsafe captures
- poor king safety
- endgame conversion
- repetition/draw handling
- tactical horizon problems
