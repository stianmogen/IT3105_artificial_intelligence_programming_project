import copy
from enum import Enum

import numpy as np


class Player(Enum):
    WHITE = 1
    BLACK = 2


class DisjointSet:
    def __init__(self, size):
        self.parent = [-1] * size
        self.rank = [0] * size

    def make_set(self, element):
        self.parent[element] = element
        self.rank[element] = 0

    def find(self, element):
        if self.parent[element] == element:
            return element
        self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, element1, element2):
        root1 = self.find(element1)
        root2 = self.find(element2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                root1, root2 = root2, root1
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

class HexGameState:
    def __init__(self, size):
        self.size = size
        self.current_player = Player.WHITE
        self.board = np.zeros((size, size), dtype=int)
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.winner = None
        self.neighbor_patterns = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
        self.empty_spaces = set((y, x) for x in range(size) for y in range(size))
        self.last_move = None
        self.dset = DisjointSet(size * size + 2) # +2 for the dummy nodes

        for i in range(size):
            for j in range(size):
                node = i * size + j + 1
                if i == 0:
                    self.dset.union(node, 0) # connect to left dummy node
                elif i == size - 1:
                    self.dset.union(node, size * size + 1) # connect to right dummy node

                for y_, x_ in self.neighbor_patterns:
                    if 0 <= i + y_ < size and 0 <= j + x_ < size:
                        neighbor = (i + y_) * size + (j + x_) + 1
                        self.dset.union(node, neighbor)

    def place_piece(self, y, x):
        if not self.board[y][x]:
            self.board[y][x] = self.current_player.value
            self.last_move = (y, x)
            self.empty_spaces.remove((y, x))
            if self.check_win(y, x):
                self.winner = self.current_player.value
            elif len(self.empty_spaces) == 0:
                self.winner = -1
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            return True
        return False

    def check_win(self, y, x):
        player = self.board[y][x]
        node = y * self.size + x + 1
        if player == Player.WHITE.value:
            return self.dset.find(node) == self.dset.find(0) and self.dset.find(node) == self.dset.find(
                self.size * self.size + 1)
        elif player == Player.BLACK.value:
            return self.dset.find(node) == self.dset.find(0) and self.dset.find(node) == self.dset.find(
                self.size * (self.size - 1) + 1)
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
