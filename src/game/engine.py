# src/game/engine.py
"""
SelfPlay-Tac game engine (Phase 1)

Features:
- Board(n, k) for n x n board with k-in-a-row win condition
- Fast-ish numpy-backed representation
- move <-> index mapping helpers
- legal_moves, play, is_draw, winner, is_game_over
- canonical tensor encoding for NN (2 planes)
- serialization (simple string)
- symmetry helpers (rotations + flips) for state and policy vectors
"""

from __future__ import annotations
from typing import List, Tuple, Iterable
import numpy as np
import copy

Move = Tuple[int, int]  # (row, col)


class Board:
    def __init__(self, n: int = 5, k: int = 4):
        assert n >= 1 and k >= 1 and k <= n
        self.n = n
        self.k = k
        # board values: 0 empty, 1 player1 (X), -1 player2 (O)
        self.board = np.zeros((n, n), dtype=np.int8)
        self.to_move = 1  # 1 or -1
        self.move_number = 0
        self._history: List[Tuple[int, int]] = []  # list of moves (r,c) in order

    # -------------------------
    # Move helpers
    # -------------------------
    def move_to_index(self, move: Move) -> int:
        r, c = move
        return int(r * self.n + c)

    def index_to_move(self, idx: int) -> Move:
        return (int(idx // self.n), int(idx % self.n))

    def legal_moves(self) -> List[Move]:
        zeros = np.argwhere(self.board == 0)
        return [tuple(pos) for pos in zeros]

    def legal_indices(self) -> List[int]:
        zeros = np.argwhere(self.board == 0)
        return [self.move_to_index(tuple(pos)) for pos in zeros]

    # -------------------------
    # Play / undo
    # -------------------------
    def play(self, move: Move | int):
        """Play a move given by (r,c) or index."""
        if isinstance(move, int):
            move = self.index_to_move(move)
        r, c = move
        if not (0 <= r < self.n and 0 <= c < self.n):
            raise ValueError("Move out of bounds")
        if self.board[r, c] != 0:
            raise ValueError("Illegal move: cell not empty")
        self.board[r, c] = self.to_move
        self._history.append((r, c))
        self.to_move *= -1
        self.move_number += 1

    def undo(self):
        """Undo last move (if any)."""
        if not self._history:
            return
        r, c = self._history.pop()
        self.board[r, c] = 0
        self.to_move *= -1
        self.move_number -= 1

    # -------------------------
    # Terminal checks
    # -------------------------
    def winner(self) -> int:
        """
        Return 1 if player1 wins, -1 if player2 wins, 0 otherwise.
        Naive but correct scan across board for k in a row.
        """
        b = self.board
        n, k = self.n, self.k
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(n):
            for c in range(n):
                val = int(b[r, c])
                if val == 0:
                    continue
                for dr, dc in dirs:
                    rr, cc = r, c
                    count = 0
                    for step in range(k):
                        nr = rr + dr * step
                        nc = cc + dc * step
                        if 0 <= nr < n and 0 <= nc < n and b[nr, nc] == val:
                            count += 1
                        else:
                            break
                    if count == k:
                        return val
        return 0

    def is_draw(self) -> bool:
        return self.winner() == 0 and not np.any(self.board == 0)

    def is_game_over(self) -> bool:
        return (self.winner() != 0) or self.is_draw()

    # -------------------------
    # Encodings and serialization
    # -------------------------
    def to_tensor(self) -> np.ndarray:
        """
        Return 2 x n x n tensor:
            plane 0 = 1 where current player's stones are
            plane 1 = 1 where opponent's stones are
        This canonicalizes perspective to current player.
        """
        cur = (self.board == self.to_move).astype(np.float32)
        opp = (self.board == -self.to_move).astype(np.float32)
        return np.stack([cur, opp], axis=0)  # shape (2, n, n)

    def serialize(self) -> str:
        """Simple serialization: n,k,boardflat,to_move"""
        flat = ",".join(map(str, self.board.flatten().tolist()))
        return f"{self.n}|{self.k}|{flat}|{int(self.to_move)}|{self.move_number}"

    @classmethod
    def deserialize(cls, s: str) -> "Board":
        n_str, k_str, flat_str, to_move_str, move_num_str = s.split("|")
        n = int(n_str); k = int(k_str)
        b = cls(n=n, k=k)
        arr = np.fromiter(map(int, flat_str.split(",")), dtype=np.int8)
        b.board = arr.reshape((n, n)).copy()
        b.to_move = int(to_move_str)
        b.move_number = int(move_num_str)
        # rebuild history not stored; that's fine for Phase 1
        return b

    # -------------------------
    # Symmetry helpers
    # -------------------------
    def copy(self) -> "Board":
        return copy.deepcopy(self)

    def apply_symmetry(self, transf: str) -> "Board":
        """
        transf: one of 'identity', 'rot90', 'rot180', 'rot270', 'flip', 'flip_rot90', 'flip_rot180', 'flip_rot270'
        Returns a new Board with transformation applied.
        """
        t = transf.lower()
        b = self.copy()
        mat = b.board.copy()
        if t == "identity":
            pass
        elif t == "rot90":
            mat = np.rot90(mat, k=1)
        elif t == "rot180":
            mat = np.rot90(mat, k=2)
        elif t == "rot270":
            mat = np.rot90(mat, k=3)
        elif t == "flip":
            mat = np.fliplr(mat)
        elif t == "flip_rot90":
            mat = np.rot90(np.fliplr(mat), k=1)
        elif t == "flip_rot180":
            mat = np.rot90(np.fliplr(mat), k=2)
        elif t == "flip_rot270":
            mat = np.rot90(np.fliplr(mat), k=3)
        else:
            raise ValueError("Unknown transform")
        b.board = mat
        # Note: to_move remains the same relative perspective, but history indices are not transformed here.
        return b

    @staticmethod
    def all_transforms() -> List[str]:
        return ["identity", "rot90", "rot180", "rot270", "flip", "flip_rot90", "flip_rot180", "flip_rot270"]

    # Transform a policy vector (flat length n*n) according to transform
    def transform_policy(self, policy: np.ndarray, transf: str) -> np.ndarray:
        assert policy.size == self.n * self.n
        mat = policy.reshape((self.n, self.n)).copy()
        t = transf.lower()
        if t == "identity":
            out = mat
        elif t == "rot90":
            out = np.rot90(mat, k=1)
        elif t == "rot180":
            out = np.rot90(mat, k=2)
        elif t == "rot270":
            out = np.rot90(mat, k=3)
        elif t == "flip":
            out = np.fliplr(mat)
        elif t == "flip_rot90":
            out = np.rot90(np.fliplr(mat), k=1)
        elif t == "flip_rot180":
            out = np.rot90(np.fliplr(mat), k=2)
        elif t == "flip_rot270":
            out = np.rot90(np.fliplr(mat), k=3)
        else:
            raise ValueError("Unknown transform")
        return out.reshape(-1)

    # convenience: produce all symmetric (state_tensor, policy_vector) pairs
    def symmetries(self, policy: np.ndarray | None = None) -> Iterable[Tuple[np.ndarray, np.ndarray | None]]:
        """
        Yields (state_tensor, policy_vector) for all 8 symmetries.
        If policy is None, yields (state_tensor, None).
        """
        base = self.board
        for t in Board.all_transforms():
            new_board = Board(n=self.n, k=self.k)
            new_board.board = getattr(np, 'array')(base)  # copy
            # apply same transform as apply_symmetry but on array
            if t == "identity":
                mat = new_board.board
            elif t == "rot90":
                mat = np.rot90(new_board.board, k=1)
            elif t == "rot180":
                mat = np.rot90(new_board.board, k=2)
            elif t == "rot270":
                mat = np.rot90(new_board.board, k=3)
            elif t == "flip":
                mat = np.fliplr(new_board.board)
            elif t == "flip_rot90":
                mat = np.rot90(np.fliplr(new_board.board), k=1)
            elif t == "flip_rot180":
                mat = np.rot90(np.fliplr(new_board.board), k=2)
            elif t == "flip_rot270":
                mat = np.rot90(np.fliplr(new_board.board), k=3)
            else:
                raise ValueError("Unknown transform")
            new_board.board = mat
            # canonicalize perspective: ensure to_move is the same as original
            new_board.to_move = self.to_move
            new_board.move_number = self.move_number
            if policy is None:
                yield new_board.to_tensor(), None
            else:
                transformed_policy = self.transform_policy(policy, t)
                yield new_board.to_tensor(), transformed_policy

    # Pretty print
    def __str__(self) -> str:
        def ch(v):
            if v == 1:
                return "X"
            if v == -1:
                return "O"
            return "."
        rows = ["".join(ch(int(x)) for x in row) for row in self.board]
        return "\n".join(rows)
