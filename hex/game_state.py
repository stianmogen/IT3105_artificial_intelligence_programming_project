import copy

import numpy as np

from disjoint_set import DisjointSet


class HexGameState:
    def __init__(self, size):
        self.size = size
        self.current_player = 1
        self.board = np.zeros((size, size), dtype=int)
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.last_move = None
        self.winner = None
        self.neighbor_patterns = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
        self.empty_spaces = set((y, x) for x in range(size) for y in range(size))
        self.left_right = DisjointSet(size * size + 2)
        self.top_bottom = DisjointSet(size * size + 2)

        for i in range(size):
            self.top_bottom.union(size * size, i)
            self.top_bottom.union(size * size + 1, size * size - 1 - i)
            self.left_right.union(size * size, i * size)
            self.left_right.union(size * size + 1, i * size + size - 1)

    def place_piece(self, y, x):
        if not self.board[y][x]:
            self.board[y][x] = self.current_player
            self.last_move = (y, x)
            self.empty_spaces.remove((y, x))
            if self.check_win(y, x):
                self.winner = self.current_player
            elif len(self.empty_spaces) == 0:
                self.winner = -1
            self.current_player = 1 if self.current_player == 2 else 2
            return True
        return False

    def neighbours(self, y, x):
        return [(y + y_, x + x_) for y_, x_ in self.neighbor_patterns if
                (0 <= (y + y_) < self.size) and (0 <= (x + x_) < self.size)]

    def check_win(self, y, x):
        player = self.board[y][x]

        if player == 1:
            for y_, x_ in self.neighbours(y, x):
                if self.board[y_][x_] == player:
                    self.left_right.union(y_ * self.size + x_, y * self.size + x)
            return self.left_right.connected(self.size * self.size, self.size * self.size + 1)
        elif player == 2:
            for y_, x_ in self.neighbours(y, x):
                if self.board[y_][x_] == player:
                    self.top_bottom.union(y_ * self.size + x_, y * self.size + x)
            return self.top_bottom.connected(self.size * self.size, self.size * self.size + 1)
        return False

    def clone_board(self):
        return copy.deepcopy(self.board)

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
