"""Rating diagnostics for external Stockfish benchmark results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from mcchess.eval.stockfish_protocol import StockfishGameRecord


@dataclass(frozen=True)
class EloEstimate:
    """Rough logistic Elo estimate against Stockfish UCI Elo-handicap levels."""

    status: str
    method: str
    included_games: int
    score: float | None
    estimated_elo: int | None
    lower_95: int | None
    upper_95: int | None
    bounded: str | None
    opponent_elo_min: int | None
    opponent_elo_max: int | None
    note: str


def estimate_mcchess_elo(games: Sequence[StockfishGameRecord]) -> EloEstimate:
    """Estimate McChess Elo against Stockfish UCI levels with a rough MLE."""

    included = [
        game
        for game in games
        if game.status == "completed"
        and game.include_in_elo
        and game.stockfish_elo is not None
        and game.mcchess_score is not None
    ]
    if not included:
        return insufficient_elo_estimate()

    opponent_elos = [game.stockfish_elo for game in included if game.stockfish_elo is not None]
    scores = [float(game.mcchess_score) for game in included if game.mcchess_score is not None]
    score = sum(scores) / len(scores)
    grid_min = max(100, min(opponent_elos) - 1000)
    grid_max = max(opponent_elos) + 1000

    likelihoods = [
        (elo, _log_likelihood(elo, opponent_elos, scores)) for elo in range(grid_min, grid_max + 1)
    ]
    estimate, best_ll = max(likelihoods, key=lambda item: item[1])
    threshold = best_ll - 1.920729410347062
    interval = [elo for elo, ll in likelihoods if ll >= threshold]
    lower_95: int | None = min(interval) if interval else estimate
    upper_95: int | None = max(interval) if interval else estimate
    bounded: str | None = None
    if estimate == grid_min:
        bounded = "lower"
        lower_95 = None
    elif estimate == grid_max:
        bounded = "upper"
        upper_95 = None

    note = (
        "Rough local estimate from Stockfish UCI_Elo handicap games. This is not "
        "Lichess Elo, FIDE Elo, or a training target."
    )
    if bounded is not None:
        note += " The estimate is bounded by the tested bracket; expand the bracket to refine it."

    return EloEstimate(
        status="ok",
        method="rough_logistic_mle_against_stockfish_uci_elo",
        included_games=len(included),
        score=score,
        estimated_elo=estimate,
        lower_95=lower_95,
        upper_95=upper_95,
        bounded=bounded,
        opponent_elo_min=min(opponent_elos),
        opponent_elo_max=max(opponent_elos),
        note=note,
    )


def insufficient_elo_estimate() -> EloEstimate:
    return EloEstimate(
        status="insufficient_data",
        method="rough_logistic_mle_against_stockfish_uci_elo",
        included_games=0,
        score=None,
        estimated_elo=None,
        lower_95=None,
        upper_95=None,
        bounded=None,
        opponent_elo_min=None,
        opponent_elo_max=None,
        note="No completed UCI_Elo games were available for Elo estimation.",
    )


def _log_likelihood(elo: int, opponent_elos: Sequence[int], scores: Sequence[float]) -> float:
    likelihood = 0.0
    for opponent_elo, score in zip(opponent_elos, scores, strict=True):
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - elo) / 400.0))
        expected = min(max(expected, 1e-12), 1.0 - 1e-12)
        likelihood += score * math.log(expected) + (1.0 - score) * math.log(1.0 - expected)
    return likelihood
