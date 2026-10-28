# ResNet-C Epoch 30: UCI 1400–3190 With Paired Openings

The fixed published epoch-30 model scored **112 wins, 46 draws, and 42 losses
over 200 games (67.5%)** against Stockfish 18 UCI handicaps. No game reached
the 400-ply cap. All 18,726 played plies were legally replayed, with matching
final positions, termination reasons, winners, and perspective scores.

This is external evaluation only. No engine moves, evaluations, or outcomes
were used for training, checkpoint selection, or search-parameter tuning.

## Protocol And Provenance

| Setting | Value |
|---|---|
| Model | [Published ResNet-C epoch 30](../models_archive/README.md), unchanged weights |
| Code | `119ca8f9c4e923df4d40828f9226cf4009a74e45` (includes final-ply outcome fix #49) |
| Search | 1,000 MCTS simulations per move, batch 8, `c_puct=1.5`, CUDA |
| Opponent | Stockfish 18, `stockfish-windows-x86-64-avx2.exe`; 1.0 second per move, 1 thread, 16 MiB hash |
| Levels | 1400–3000 in steps of 200, then 3190 (the installed binary's maximum) |
| Schedule | 20 games per level; each of 10 fixed FENs played with both colors |
| Seed | 0; Stockfish randomness and time-limited searches are not fully deterministic |
| Draw rules | `python-chess` outcomes with claimable draws enabled; nonterminal draw at 400 played plies |
| Started / finished | 2026-10-28 09:53:20 / 16:45:38 UTC |
| Runtime | 24,738.31 seconds (6 h 52 min 18 s) |
| Hardware | Ryzen 7 7735HS, RTX 4060 Laptop GPU (8 GiB), driver 566.36, Windows 11 |
| Software | Python 3.12.5, PyTorch 2.11.0+cu126, CUDA 12.6, python-chess 1.11.2 |

The ten opening lines were manually fixed before this run, independently of
engine evaluation or benchmark outcomes. Each contains six legal opening plies;
games start from the resulting FEN, without the preceding move stack. The
400-ply cap counts moves after that FEN. Exact SAN/UCI lines, FENs, hardware,
software, dataset-manifest reference, and engine defaults are preserved in the
[result JSON](data/stockfish_resnet_c_epoch30_1400_3190_paired10.json).
This is a small opening set, not a representative or engine-balanced corpus.

## Results

Every level completed 20 games. Scores are from McChess's perspective.

| Stockfish `UCI_Elo` | Wins | Draws | Losses | Score |
|---:|---:|---:|---:|---:|
| 1400 | 20 | 0 | 0 | 100.0% |
| 1600 | 20 | 0 | 0 | 100.0% |
| 1800 | 18 | 1 | 1 | 92.5% |
| 2000 | 16 | 2 | 2 | 85.0% |
| 2200 | 17 | 2 | 1 | 90.0% |
| 2400 | 11 | 6 | 3 | 70.0% |
| 2600 | 8 | 8 | 4 | 60.0% |
| 2800 | 2 | 11 | 7 | 37.5% |
| 3000 | 0 | 7 | 13 | 17.5% |
| 3190 | 0 | 9 | 11 | 22.5% |

With White: 58 W / 27 D / 15 L, 71.5%. With Black: 54 W / 19 D / 27 L, 63.5%.
Scores aggregated over levels ranged from 60% to 70% across the ten openings;
there are only two games per opening at each level.

All 154 decisive games ended by checkmate. Draws were 36 by claimable repetition,
four by insufficient material, four by stalemate, and two by the fifty-move
rule. All nine draws at 3190 were repetitions. Longest game: 314 played plies;
illegal moves and move-cap draws: **0**.

The runner estimates **2653**, with a rough 95% logistic-model interval of
**2571–2736**. This describes the configured Stockfish handicap benchmark,
not FIDE, Lichess, CCRL, or general engine strength. The interval does not cover
all dependencies from repeated openings or engine state, or uncertainty in the
handicap calibration. The higher score at 3190 than at 3000 is an observed
small-sample result, not evidence that the levels are reversed.

The model is identical to the [August run](2026-08-12-stockfish-mcts1000-resnet-c-epoch30-200games.md).
Different openings, opponent levels, and move caps prevent an improvement claim
from comparing 2653 with its earlier 2480 estimate. No full-strength sanity
games were included in this October schedule; the earlier full-strength losses
remain documented.

## Preceding 12-Game Probe (October 27)

The [probe JSON](data/stockfish_resnet_c_epoch30_2400_2800_probe.json) preserves
all 12 games from the standard position, with a 180-ply cap and the same model,
search budget, engine binary, hardware, and one-second opponent limit. It ran
from 22:02:11 to 22:33:52 UTC (1,901.78 seconds), using the same code commit.

| Stockfish `UCI_Elo` | Wins | Draws | Losses | Score | Move-cap draws |
|---:|---:|---:|---:|---:|---:|
| 2400 | 2 | 2 | 0 | 75.0% | 0 |
| 2600 | 2 | 1 | 1 | 62.5% | 0 |
| 2800 | 0 | 3 | 1 | 37.5% | 2 |

Total: 4 W / 6 D / 2 L, 58.33%. Games 9 and 11 were nonterminal at the cap and
received 0.5 under the recorded protocol. All 1,340 plies replay legally. The
saved estimate, 2671 (2453–2899), is retained as a small-sample diagnostic.
[Issue #50](https://github.com/MecGlandorff/McChess/issues/50) tracks separate
reporting of capped games; it does not require correcting the 200-game result.

## Reproduction And Artifact Handling

Use the [full config](../configs/eval/stockfish_mcts1000_resnet_c_epoch30_batch8_1400_3190_paired10.yaml)
or [probe config](../configs/eval/stockfish_mcts1000_resnet_c_epoch30_batch8_2400_2800_probe.yaml).
Set `STOCKFISH_PATH` to the installed binary. The configs use separate rerun
destinations; choose a fresh `run_id` and `output_dir` before each invocation:

```powershell
poetry run python -m mcchess.eval.stockfish PATH_TO_NEW_CONFIG --show --keep-awake
```

The [model manifest](../models_archive/manifest.json) records both published JSON
checksums. Each JSON retains every game and original score, adds saved provenance,
and records publication transformations. Only machine-specific engine parent
directories and the absolute config-path prefix were removed from source result
fields. Raw runs remain unchanged under ignored `runs/`.

- Model SHA-256: `17ee11db2d007ff26f0842c561d42d6e53e3d1f7c042da8b9ba0d8d42b6af02f`.
- Engine SHA-256: `c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911`.
- Source full-result SHA-256: `473ac1f880e317ca917682c2c7b5555b8a6da51cf55a6358d2fbab30fef84f78`.
- Source probe-result SHA-256: `801e92c4bbfd7048f9b2a102b3b0c0512f6acab0243fe35f303280a842128877`.
