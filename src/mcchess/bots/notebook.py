"""Notebook helpers for playing against bots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import chess.svg
import ipywidgets as widgets  # type: ignore[import-untyped]

from mcchess.bots.base import Bot

BOARD_SVG_SIZE = 390
BOARD_SQUARE_SIZE = "52px"
LIGHT_SQUARE_COLOR = "#e8c99b"
DARK_SQUARE_COLOR = "#9b673c"
LAST_MOVE_LIGHT_COLOR = "#d9b26e"
LAST_MOVE_DARK_COLOR = "#b9823f"
SELECTED_SQUARE_COLOR = "#f5cf55"
WHITE_PIECE_COLOR = "#fff7e6"
BLACK_PIECE_COLOR = "#1f1712"
BOARD_WIDGET_CSS = """
<style>
.mcchess-board-square button {
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: inset 0 0 0 1px rgba(31, 23, 18, 0.18);
    font-family: "Segoe UI Symbol", "Noto Sans Symbols2", "DejaVu Sans", sans-serif !important;
    line-height: 1 !important;
    padding: 0 !important;
}
.mcchess-white-piece button {
    text-shadow: 0 1px 2px rgba(31, 23, 18, 0.85), 0 0 1px rgba(31, 23, 18, 0.95);
}
.mcchess-black-piece button {
    text-shadow: 0 1px 1px rgba(255, 247, 230, 0.6);
}
</style>
"""


@dataclass
class NotebookChessGame:
    """Human-vs-bot game state used by the notebook widget.

    :meth:`play` remains available for quick SAN or UCI debugging from a cell.
    """

    bot: Bot
    human_color: chess.Color = chess.WHITE
    board: chess.Board = field(default_factory=chess.Board)
    status: str = field(init=False, default="Your move.")

    def __post_init__(self) -> None:
        if self.board.turn != self.human_color and not self.board.is_game_over(claim_draw=True):
            self.status = self._bot_move()

    def play(self, move_text: str) -> "NotebookChessGame":
        """Apply one human move and answer with the bot's reply."""

        if self.board.is_game_over(claim_draw=True):
            self.status = self._game_over_message()
        else:
            try:
                move = self._parse_move(move_text)
            except ValueError:
                self.status = f"Not a legal move here: {move_text}"
            else:
                self.apply_human_move(move)
        print(self.status)
        return self

    def apply_human_move(self, move: chess.Move, *, reply: bool = True) -> bool:
        """Push a legal human move and optionally let the bot reply.

        Returns ``True`` when the human move was applied. The clickable widget
        uses ``reply=False`` so it can repaint the human move before a slower
        MCTS reply starts.
        """

        if self.board.is_game_over(claim_draw=True):
            self.status = self._game_over_message()
            return False
        if move not in self.board.legal_moves:
            self.status = f"Illegal move: {move.uci()}"
            return False

        played = f"You played {self.board.san(move)}."
        self.board.push(move)
        if self.board.is_game_over(claim_draw=True):
            self.status = f"{played} {self._game_over_message()}"
        elif reply:
            self.status = f"{played} {self._bot_move()}"
        else:
            self.status = f"{played} Bot thinking."
        return True

    def apply_bot_move(self) -> None:
        """Let the bot move from the current board position."""

        if self.board.is_game_over(claim_draw=True):
            self.status = self._game_over_message()
            return
        if self.board.turn == self.human_color:
            self.status = "Your move."
            return
        self.status = self._bot_move()

    def _parse_move(self, move_text: str) -> chess.Move:
        try:
            return self.board.parse_san(move_text)
        except ValueError:
            return self.board.parse_uci(move_text)

    def _bot_move(self) -> str:
        move = self.bot.choose_move(self.board.copy(stack=False))
        if move not in self.board.legal_moves:
            raise ValueError(f"bot returned illegal move {move.uci()}")
        text = f"Bot played {self.board.san(move)}."
        self.board.push(move)
        if self.board.is_game_over(claim_draw=True):
            text = f"{text} {self._game_over_message()}"
        return text

    def _game_over_message(self) -> str:
        outcome = self.board.outcome(claim_draw=True)
        termination = outcome.termination.name.lower() if outcome is not None else "unknown"
        return f"Game over: {self.board.result(claim_draw=True)} ({termination})."

    def _repr_svg_(self) -> str:
        return chess.svg.board(
            self.board,
            orientation=self.human_color,
            lastmove=self.board.peek() if self.board.move_stack else None,
            check=self.board.king(self.board.turn) if self.board.is_check() else None,
            size=BOARD_SVG_SIZE,
        )


class ClickableChessBoard:
    """Click-to-move ipywidgets view over a :class:`NotebookChessGame`.

    The 64 square buttons are created once and mutated in place on refresh,
    so the board never rebuilds (or flickers) between moves.
    """

    def __init__(self, game: NotebookChessGame) -> None:
        self.game = game
        self._selected: chess.Square | None = None
        self._status = widgets.HTML()
        self._buttons: dict[chess.Square, widgets.Button] = {}
        for square in self._display_squares():
            button = widgets.Button(
                layout=widgets.Layout(width=BOARD_SQUARE_SIZE, height=BOARD_SQUARE_SIZE),
            )
            button.add_class("mcchess-board-square")
            button.on_click(self._make_click_handler(square))
            self._buttons[square] = button
        board_grid = widgets.GridBox(
            tuple(self._buttons.values()),
            layout=widgets.Layout(
                border="3px solid #5c3922",
                grid_template_columns=f"repeat(8, {BOARD_SQUARE_SIZE})",
                grid_template_rows=f"repeat(8, {BOARD_SQUARE_SIZE})",
                grid_gap="0",
            ),
        )
        self.widget = widgets.VBox([self._status, board_grid, widgets.HTML(value=BOARD_WIDGET_CSS)])
        self._refresh()

    def click_square(self, square: str | chess.Square) -> None:
        """Handle a source/target square click, also useful in tests."""

        clicked = chess.parse_square(square) if isinstance(square, str) else square
        board = self.game.board
        if board.is_game_over(claim_draw=True):
            self._refresh()
            return

        piece = board.piece_at(clicked)
        if piece is not None and piece.color == self.game.human_color == board.turn:
            self._selected = clicked
            self.game.status = f"Selected {chess.square_name(clicked)}."
        elif self._selected is None:
            name = chess.square_name(clicked)
            self.game.status = f"No piece on {name}." if piece is None else f"Cannot move piece on {name}."
        else:
            move = self._build_move(self._selected, clicked)
            self._selected = None
            moved = self.game.apply_human_move(move, reply=False)
            if moved and not self.game.board.is_game_over(claim_draw=True):
                self._refresh()
                self.game.apply_bot_move()
        self._refresh()

    def _display_squares(self) -> list[chess.Square]:
        if self.game.human_color == chess.WHITE:
            ranks, files = range(7, -1, -1), range(8)
        else:
            ranks, files = range(8), range(7, -1, -1)
        return [chess.square(file, rank) for rank in ranks for file in files]

    def _make_click_handler(self, square: chess.Square) -> Callable[[widgets.Button], None]:
        def handle_click(_: widgets.Button) -> None:
            self.click_square(square)

        return handle_click

    def _build_move(self, source: chess.Square, target: chess.Square) -> chess.Move:
        piece = self.game.board.piece_at(source)
        promotion = None
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(target) in (0, 7):
            promotion = chess.QUEEN
        return chess.Move(source, target, promotion=promotion)

    def _refresh(self) -> None:
        for square, button in self._buttons.items():
            piece = self.game.board.piece_at(square)
            self._set_piece_classes(button, piece)
            with button.hold_trait_notifications():
                # A single space forces a repaint on vacated squares while
                # rendering blank; an empty description can leave stale glyphs.
                button.description = piece.unicode_symbol() if piece is not None else " "
                button.icon = ""
                button.tooltip = chess.square_name(square)
                button.style.button_color = self._square_color(square)
                button.style.font_size = "34px"
                button.style.font_weight = "600"
                button.style.text_color = self._piece_color(piece)
        turn = "White" if self.game.board.turn == chess.WHITE else "Black"
        self._status.value = f"<b>{turn} to move.</b> {self.game.status}"

    def _square_color(self, square: chess.Square) -> str:
        if square == self._selected:
            return SELECTED_SQUARE_COLOR

        last_move_squares: set[chess.Square] = set()
        if self.game.board.move_stack:
            last_move = self.game.board.peek()
            last_move_squares = {last_move.from_square, last_move.to_square}

        is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        if square in last_move_squares:
            return LAST_MOVE_LIGHT_COLOR if is_light else LAST_MOVE_DARK_COLOR
        return LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR

    def _piece_color(self, piece: chess.Piece | None) -> str:
        if piece is None:
            return BLACK_PIECE_COLOR
        return WHITE_PIECE_COLOR if piece.color == chess.WHITE else BLACK_PIECE_COLOR

    def _set_piece_classes(self, button: widgets.Button, piece: chess.Piece | None) -> None:
        button.remove_class("mcchess-white-piece")
        button.remove_class("mcchess-black-piece")
        if piece is None:
            return
        if piece.color == chess.WHITE:
            button.add_class("mcchess-white-piece")
        else:
            button.add_class("mcchess-black-piece")

    def _ipython_display_(self) -> None:
        from IPython.display import display  # type: ignore[import-untyped]

        display(self.widget)
