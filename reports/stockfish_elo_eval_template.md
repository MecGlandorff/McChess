# Stockfish External Benchmark Report Template

Status: pending report-scale rerun from committed code.

Scope: This is an external evaluation benchmark only. Do not use Stockfish
moves, evaluations, or game outcomes for McChess training data, labels,
distillation targets, or checkpoint selection targets.

## Run Metadata

- Date:
- Git commit:
- Config: `configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml`
- Command:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
```

- Checkpoint:
- Stockfish binary:
- Hardware:
- Output directory:

## Protocol

- Agent: ResNet-B MCTS-200
- Opponent: Stockfish UCI
- Opening protocol: standard initial position
- Color policy: alternating McChess White first
- Max ply: 180
- Draw rule: python-chess outcome or max-ply draw
- Full-strength sanity games: excluded from Elo estimate
- UCI Elo games: included in Elo estimate when completed

## Results

| Games | Wins | Draws | Losses | Score | Illegal moves |
|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Elo Estimate

| Included games | Opponent Elo range | Estimated Elo | Rough interval | Bounded |
|---:|---|---:|---|---|
| TBD | TBD | TBD | TBD | TBD |

Interpretation: the number above is a local Stockfish-UCI benchmark estimate
under the recorded config. It is not Lichess Elo, FIDE Elo, or a general engine
strength claim.

## Game Table

| Game | Stockfish level | White | Black | Result | Winner | Winner name | McChess score | Included in Elo |
|---:|---|---|---|---|---|---|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Notes

- Result status:
- Failures or exclusions:
- Follow-up:
