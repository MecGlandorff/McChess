"""Terminal play against a policy-only or fixed-budget MCTS checkpoint bot."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import chess

from mcchess.bots import Bot, MCTSBot, PolicyOnlyBot

PlayMode = Literal["policy", "mcts"]
ColorName = Literal["white", "black"]

CANONICAL_MODEL_RELATIVE_PATH = Path("models_archive") / "resnet_c_epoch_030.pt"
DEFAULT_MCTS_SIMULATIONS = 800
DEFAULT_C_PUCT = 1.5
DEFAULT_INFERENCE_BATCH_SIZE = 1
EXIT_COMMANDS = frozenset({"exit", "quit", "resign"})


def canonical_model_path() -> Path:
    """Resolve the bundled model from a repository checkout."""

    checkout_root = Path(__file__).resolve().parents[2]
    candidates = (
        Path.cwd() / CANONICAL_MODEL_RELATIVE_PATH,
        checkout_root / CANONICAL_MODEL_RELATIVE_PATH,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1]


def parse_human_move(board: chess.Board, move_text: str) -> chess.Move:
    """Parse one legal SAN or UCI move in the current position."""

    normalized = move_text.strip()
    if not normalized:
        raise ValueError("move must not be empty")
    try:
        return board.parse_san(normalized)
    except ValueError:
        try:
            return board.parse_uci(normalized.lower())
        except ValueError as uci_error:
            raise ValueError(f"not a legal SAN or UCI move: {move_text}") from uci_error


def format_board(board: chess.Board, *, orientation: chess.Color) -> str:
    """Render an ASCII board from the human player's orientation."""

    ranks = range(7, -1, -1) if orientation == chess.WHITE else range(8)
    files = range(8) if orientation == chess.WHITE else range(7, -1, -1)
    rows = []
    for rank in ranks:
        squares = []
        for file_index in files:
            piece = board.piece_at(chess.square(file_index, rank))
            squares.append(piece.symbol() if piece is not None else ".")
        rows.append(f"{rank + 1}  {' '.join(squares)}")
    file_labels = " ".join(chess.FILE_NAMES[file_index] for file_index in files)
    rows.append(f"   {file_labels}")
    return "\n".join(rows)


def build_play_bot(
    checkpoint_path: str | Path,
    *,
    mode: PlayMode,
    device: str,
    simulations: int = DEFAULT_MCTS_SIMULATIONS,
    c_puct: float = DEFAULT_C_PUCT,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> PolicyOnlyBot | MCTSBot:
    """Load the requested terminal-play bot."""

    if mode == "policy":
        return PolicyOnlyBot.from_checkpoint(
            checkpoint_path,
            device=device,
            name="mcchess_policy",
        )
    if mode == "mcts":
        return MCTSBot.from_checkpoint(
            checkpoint_path,
            device=device,
            name=f"mcchess_mcts_{simulations}",
            simulations=simulations,
            c_puct=c_puct,
            inference_batch_size=inference_batch_size,
        )
    raise ValueError(f"unsupported play mode: {mode}")


def play_terminal_game(
    bot: Bot,
    *,
    human_color: chess.Color = chess.WHITE,
    board: chess.Board | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> chess.Outcome | None:
    """Play one terminal game, returning ``None`` when the human exits early."""

    game = board.copy(stack=True) if board is not None else chess.Board()
    output_fn(format_board(game, orientation=human_color))

    while not game.is_game_over(claim_draw=True):
        if game.turn == human_color:
            try:
                move_text = input_fn("Your move (SAN or UCI; 'quit' to stop): ").strip()
            except (EOFError, KeyboardInterrupt):
                output_fn("Game stopped.")
                return None
            if move_text.lower() in EXIT_COMMANDS:
                output_fn("Game stopped.")
                return None
            try:
                move = parse_human_move(game, move_text)
            except ValueError as exc:
                output_fn(str(exc))
                continue
            san = game.san(move)
            game.push(move)
            output_fn(f"You played {san} ({move.uci()}).")
        else:
            output_fn(f"{bot.name} is thinking...")
            move = bot.choose_move(game.copy(stack=True))
            if move not in game.legal_moves:
                raise ValueError(f"bot returned illegal move {move.uci()}")
            san = game.san(move)
            game.push(move)
            output_fn(f"{bot.name} played {san} ({move.uci()}).")
        output_fn(format_board(game, orientation=human_color))

    outcome = game.outcome(claim_draw=True)
    termination = outcome.termination.name.lower() if outcome is not None else "unknown"
    output_fn(f"Game over: {game.result(claim_draw=True)} ({termination}).")
    return outcome


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play against the bundled McChess model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to the bundled epoch-30 model.",
    )
    parser.add_argument(
        "--mode",
        choices=("policy", "mcts"),
        default="mcts",
        help="Move selection mode. Default: mcts.",
    )
    parser.add_argument(
        "--color",
        choices=("white", "black"),
        default="white",
        help="Human color. Default: white.",
    )
    parser.add_argument("--device", default="auto", help="Torch device. Default: auto.")
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=DEFAULT_MCTS_SIMULATIONS,
        help=f"MCTS simulations per move. Default: {DEFAULT_MCTS_SIMULATIONS}.",
    )
    parser.add_argument(
        "--c-puct",
        type=_positive_float,
        default=DEFAULT_C_PUCT,
        help=f"PUCT exploration constant. Default: {DEFAULT_C_PUCT}.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=_positive_int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help=f"MCTS leaf inference batch size. Default: {DEFAULT_INFERENCE_BATCH_SIZE}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint or canonical_model_path()
    if not checkpoint_path.is_file():
        print(f"checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    try:
        bot = build_play_bot(
            checkpoint_path,
            mode=args.mode,
            device=args.device,
            simulations=args.simulations,
            c_puct=args.c_puct,
            inference_batch_size=args.inference_batch_size,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"could not load checkpoint: {exc}", file=sys.stderr)
        return 2

    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    metadata = bot.checkpoint.metadata
    device = bot.device
    print(f"checkpoint: {metadata.path}")
    print(f"epoch: {metadata.epoch}")
    print(f"device: {device}")
    if args.mode == "mcts":
        print(
            "search: "
            f"mcts simulations={args.simulations} c_puct={args.c_puct} "
            f"inference_batch_size={args.inference_batch_size}"
        )
    else:
        print("search: policy-only")
    play_terminal_game(bot, human_color=human_color)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
