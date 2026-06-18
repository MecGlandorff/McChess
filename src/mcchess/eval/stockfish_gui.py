"""Tkinter viewer for live Stockfish benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import chess

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError as exc:  # pragma: no cover - depends on local Python build.
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR: ImportError | None = exc
else:
    _TK_IMPORT_ERROR = None

if TYPE_CHECKING:
    from mcchess.eval.stockfish import StockfishGameRecord

LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"
FROM_HIGHLIGHT = "#f6f669"
TO_HIGHLIGHT = "#baca44"
SQUARE_SIZE = 58
BOARD_SIZE = SQUARE_SIZE * 8


class StockfishEvalViewer:
    """Small two-window Tk viewer for the terminal Stockfish evaluator."""

    def __init__(self) -> None:
        if tk is None or ttk is None:
            raise RuntimeError(
                "tkinter is not available in this Python environment; run without --show."
            ) from _TK_IMPORT_ERROR

        self._tk: Any = tk
        self._ttk: Any = ttk
        self._tcl_error: type[BaseException] = tk.TclError
        self._closed = False

        self.root: Any = tk.Tk()
        self.root.title("McChess Stockfish live board")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var: Any = tk.StringVar(value="Waiting for the first move.")
        self.canvas: Any = tk.Canvas(
            self.root,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            highlightthickness=0,
        )
        self.status_label: Any = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padding=(8, 6),
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.status_label.grid(row=1, column=0, sticky="ew")

        self.table_window: Any = tk.Toplevel(self.root)
        self.table_window.title("McChess Stockfish results")
        self.table_window.protocol("WM_DELETE_WINDOW", self.close)
        self.tree: Any = self._build_table(self.table_window)

        self._draw_board(chess.Board(), last_move_uci=None)
        self._safe_update()

    def on_move(self, event: Mapping[str, Any]) -> None:
        """Update the live board from a move callback event."""

        if self._closed:
            return
        board = chess.Board(str(event["fen"]))
        last_move = str(event.get("uci") or "")
        self._draw_board(board, last_move_uci=last_move or None)
        self.status_var.set(
            "Game {game} ply {ply}: {level} {color} {bot} played {san} ({uci})".format(
                game=int(event["game_index"]) + 1,
                ply=int(event["ply"]),
                level=event["level"],
                color=event["color"],
                bot=event["bot"],
                san=event["san"],
                uci=event["uci"],
            )
        )
        self._safe_update()

    def on_game(self, game: StockfishGameRecord) -> None:
        """Update the cumulative results table after a game finishes."""

        if self._closed:
            return
        board = chess.Board(game.final_fen)
        last_move = game.moves[-1] if game.moves else None
        self._draw_board(board, last_move_uci=last_move)
        self.status_var.set(
            f"Game {game.game_index + 1} finished: {game.result} "
            f"({game.termination}); McChess score {game.mcchess_score}"
        )
        self._upsert_game_row(game)
        self._safe_update()

    def mark_complete(self, result_path: str) -> None:
        """Show the final artifact path before the user closes the windows."""

        if self._closed:
            return
        self.status_var.set(f"Run complete. Result written to {result_path}. Close windows to exit.")
        self._safe_update()

    def wait_until_closed(self) -> None:
        """Block until the user closes the viewer windows."""

        if self._closed:
            return
        try:
            self.root.mainloop()
        except self._tcl_error:
            self._closed = True

    def close(self) -> None:
        """Close both viewer windows."""

        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except self._tcl_error:
            pass

    def _build_table(self, parent: Any) -> Any:
        columns = [
            "game",
            "level",
            "white",
            "black",
            "result",
            "winner",
            "winner_name",
            "mcchess_score",
            "included",
        ]
        tree = self._ttk.Treeview(parent, columns=columns, show="headings", height=20)
        headings = {
            "game": "Game",
            "level": "Stockfish level",
            "white": "White",
            "black": "Black",
            "result": "Result",
            "winner": "Winner",
            "winner_name": "Winner name",
            "mcchess_score": "McChess score",
            "included": "Included in Elo",
        }
        widths = {
            "game": 60,
            "level": 170,
            "white": 150,
            "black": 150,
            "result": 90,
            "winner": 90,
            "winner_name": 150,
            "mcchess_score": 110,
            "included": 110,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w", stretch=True)

        yscroll = self._ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        return tree

    def _draw_board(self, board: chess.Board, *, last_move_uci: str | None) -> None:
        self.canvas.delete("all")
        highlighted = _highlighted_squares(last_move_uci)
        for rank in range(7, -1, -1):
            for file_index in range(8):
                square = chess.square(file_index, rank)
                row = 7 - rank
                col = file_index
                x0 = col * SQUARE_SIZE
                y0 = row * SQUARE_SIZE
                x1 = x0 + SQUARE_SIZE
                y1 = y0 + SQUARE_SIZE
                color = LIGHT_SQUARE if (rank + file_index) % 2 else DARK_SQUARE
                if square == highlighted[0]:
                    color = FROM_HIGHLIGHT
                elif square == highlighted[1]:
                    color = TO_HIGHLIGHT
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

                piece = board.piece_at(square)
                if piece is None:
                    continue
                self.canvas.create_text(
                    x0 + SQUARE_SIZE / 2,
                    y0 + SQUARE_SIZE / 2,
                    text=piece.unicode_symbol(),
                    fill="#111827",
                    font=("Segoe UI Symbol", 34),
                )

    def _upsert_game_row(self, game: StockfishGameRecord) -> None:
        iid = str(game.game_index)
        score = "" if game.mcchess_score is None else f"{game.mcchess_score:.1f}"
        values = (
            game.game_index + 1,
            game.level,
            game.white,
            game.black,
            game.result,
            game.winner or "draw",
            game.winner_name or "draw",
            score,
            "yes" if game.include_in_elo else "no",
        )
        if self.tree.exists(iid):
            self.tree.item(iid, values=values)
        else:
            self.tree.insert("", "end", iid=iid, values=values)
        self.tree.see(iid)

    def _safe_update(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tcl_error:
            self._closed = True


def _highlighted_squares(last_move_uci: str | None) -> tuple[chess.Square | None, chess.Square | None]:
    if not last_move_uci:
        return None, None
    try:
        move = chess.Move.from_uci(last_move_uci)
    except ValueError:
        return None, None
    return move.from_square, move.to_square
