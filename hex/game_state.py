import copy
from enum import Enum

import numpy as np


class Player(Enum):
    WHITE = 1
    BLACK = 2


class GameState:
    def __init__(self, size):
        self.size = size
        self.current_player = Player.WHITE
        self.board = np.zeros((size, size), dtype=int)
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.winner = None
        self.neighbor_patterns = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
        self.empty_spaces = set((y, x) for x in range(size) for y in range(size))

    def place_piece(self, y, x):
        if not self.board[y][x]:
            self.board[y][x] = self.current_player.value
            self.empty_spaces.remove((y, x))
            if self.check_win(y, x):
                self.winner = self.current_player.value
            elif len(self.empty_spaces) == 0:
                self.winner = -1
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            return True
        return False

    def neighbours(self, y, x):
        return [(y + y_, x + x_) for y_, x_ in self.neighbor_patterns if
                (0 <= (y + y_) < self.size) and (0 <= (x + x_) < self.size)]

    def reach_left(self, y, x, blocked):
        if x == 0:
            return True

        blocked[y][x] = True
        for y_, x_ in self.neighbours(y, x):
            if not blocked[y_][x_] and self.reach_left(y_, x_, blocked):
                return True

        return False

    def reach_right(self, y, x, blocked):
        if x == len(blocked[0]) - 1:
            return True

        blocked[y][x] = True
        for y_, x_ in self.neighbours(y, x):
            if not blocked[y_][x_] and self.reach_right(y_, x_, blocked):
                return True

        return False

    def reach_top(self, y, x, blocked):
        if y == 0:
            return True

        blocked[y][x] = True
        for y_, x_ in self.neighbours(y, x):
            if not blocked[y_][x_] and self.reach_top(y_, x_, blocked):
                return True

        return False

    def reach_bottom(self, y, x, blocked):
        if y == len(blocked) - 1:
            return True

        blocked[y][x] = True
        for y_, x_ in self.neighbours(y, x):
            if not blocked[y_][x_] and self.reach_bottom(y_, x_, blocked):
                return True

        return False

    def check_win(self, y, x):
        player = self.board[y][x]
        blocked = [[False if self.board[i][j] == player else True for j in range(self.size)] for i in range(self.size)]

        if player == Player.WHITE.value:
            return self.reach_left(y, x, list(blocked)) and self.reach_right(y, x, list(blocked))
        elif player == Player.BLACK.value:
            return self.reach_top(y, x, list(blocked)) and self.reach_bottom(y, x, list(blocked))
        return False

    def print_board(self):
        rows = self.size
        cols = self.size
        indent = 0
        headings = " " * 5 + (" " * 3).join(self.column_names[:cols])
        print(headings)
        tops = " " * 5 + (" " * 3).join("-" * cols)
        print(tops)
        roof = " " * 4 + "/ \\" + "_/ \\" * (cols - 1)
        print(roof)
        color_mapping = lambda i: " WB"[i]
        for r in range(rows):
            row_mid = " " * indent
            row_mid += " {} | ".format(r + 1)
            row_mid += " | ".join(map(color_mapping, self.board[r]))
            row_mid += " | {} ".format(r + 1)
            print(row_mid)
            row_bottom = " " * indent
            row_bottom += " " * 3 + " \\_/" * cols
            if r < rows - 1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " " * (indent - 2) + headings
        print(headings)
