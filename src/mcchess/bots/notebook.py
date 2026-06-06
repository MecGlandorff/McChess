"""Notebook helpers for playing against bots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import chess
import ipywidgets as widgets  # type: ignore[import-untyped]

from mcchess.bots.base import Bot, NoLegalMoveError


@dataclass
class NotebookChessGame:
    """Clickable notebook chess board for human-vs-bot games."""

    bot: Bot
    human_color: chess.Color = chess.WHITE
    board: chess.Board = field(default_factory=chess.Board)
    widget: widgets.Widget = field(init=False)
    _selected_square: chess.Square | None = field(init=False, default=None)
    _buttons: dict[chess.Square, widgets.Button] = field(init=False, default_factory=dict)
    _status: widgets.HTML = field(init=False)
    _board_grid: widgets.GridBox = field(init=False)

    def __post_init__(self) -> None:
        self._status = widgets.HTML()
        self._board_grid = widgets.GridBox(
            layout=widgets.Layout(
                grid_template_columns="repeat(8, 40px)",
                grid_template_rows="repeat(8, 40px)",
                grid_gap="1px",
            )
        )
        self.widget = widgets.VBox([self._status, self._board_grid])
        self._build_buttons()
        self._render()
        if self.board.turn != self.human_color:
            self.bot_move()

    def click_square(self, square: str | chess.Square) -> None:
        """Handle a source/target square click, useful in tests and notebooks."""

        clicked = chess.parse_square(square) if isinstance(square, str) else square
        if self.board.is_game_over(claim_draw=True):
            self._set_status("Game is over.")
            return
        if self.board.turn != self.human_color:
            self._set_status("Waiting for bot move.")
            return

        if self._selected_square is None:
            self._select_source(clicked)
            return

        source = self._selected_square
        self._selected_square = None
        move = self._build_human_move(source, clicked)
        if move not in self.board.legal_moves:
            self._set_status(f"Illegal move: {move.uci()}")
            self._render()
            return

        self.board.push(move)
        self._set_status(f"Human played {move.uci()}.")
        self._render()
        if not self.board.is_game_over(claim_draw=True):
            self.bot_move()

    def bot_move(self) -> None:
        """Ask the bot to move once if it is the bot's turn."""

        if self.board.is_game_over(claim_draw=True):
            self._set_game_over_status()
            self._render()
            return
        if self.board.turn == self.human_color:
            return

        try:
            move = self.bot.choose_move(self.board.copy(stack=False))
        except NoLegalMoveError:
            self._set_game_over_status()
            self._render()
            return

        if move not in self.board.legal_moves:
            raise ValueError(f"bot returned illegal move {move.uci()}")
        self.board.push(move)
        self._set_status(f"Bot played {move.uci()}.")
        self._render()

    def _build_buttons(self) -> None:
        children = []
        for square in self._display_squares():
            button = widgets.Button(layout=widgets.Layout(width="40px", height="40px"))
            button.on_click(self._make_click_handler(square))
            self._buttons[square] = button
            children.append(button)
        self._board_grid.children = tuple(children)

    def _display_squares(self) -> Iterable[chess.Square]:
        if self.human_color == chess.WHITE:
            ranks = range(7, -1, -1)
            files = range(8)
        else:
            ranks = range(8)
            files = range(7, -1, -1)

        for rank in ranks:
            for file in files:
                yield chess.square(file, rank)

    def _make_click_handler(self, square: chess.Square):
        def handle_click(_: widgets.Button) -> None:
            self.click_square(square)

        return handle_click

    def _select_source(self, square: chess.Square) -> None:
        piece = self.board.piece_at(square)
        if piece is None:
            self._set_status(f"No piece on {chess.square_name(square)}.")
            return
        if piece.color != self.human_color or piece.color != self.board.turn:
            self._set_status(f"Cannot move piece on {chess.square_name(square)}.")
            return

        self._selected_square = square
        self._set_status(f"Selected {chess.square_name(square)}.")
        self._render()

    def _build_human_move(self, source: chess.Square, target: chess.Square) -> chess.Move:
        piece = self.board.piece_at(source)
        promotion = None
        if piece is not None and piece.piece_type == chess.PAWN:
            target_rank = chess.square_rank(target)
            if target_rank == 0 or target_rank == 7:
                promotion = chess.QUEEN
                self._set_status("Promotion defaults to queen.")
        return chess.Move(source, target, promotion=promotion)

    def _render(self) -> None:
        for square, button in self._buttons.items():
            piece = self.board.piece_at(square)
            button.description = piece.unicode_symbol() if piece is not None else ""
            button.tooltip = chess.square_name(square)
            color = "#f0d9b5" if (chess.square_file(square) + chess.square_rank(square)) % 2 else "#b58863"
            if square == self._selected_square:
                color = "#facc15"
            button.style.button_color = color

        if self.board.is_game_over(claim_draw=True):
            self._set_game_over_status()

    def _set_status(self, message: str) -> None:
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        self._status.value = f"<b>{turn} to move.</b> {message}"

    def _set_game_over_status(self) -> None:
        outcome = self.board.outcome(claim_draw=True)
        result = self.board.result(claim_draw=True)
        termination = outcome.termination.name.lower() if outcome is not None else "unknown"
        self._status.value = f"<b>Game over.</b> result={result}, termination={termination}"


def create_notebook_game(
    bot: Bot,
    *,
    human_color: chess.Color = chess.WHITE,
) -> widgets.Widget:
    """Return a clickable notebook widget for playing a bot."""

    return NotebookChessGame(bot=bot, human_color=human_color).widget
