import copy

import numpy as np

from disjoint_set import DisjointSet


class HexGameState:
    def __init__(self, size):
        self.size = size
        self.current_player = 1
        self.board = np.zeros(size * size, dtype=int)
        self.empty_spaces = set(range(size * size))
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.last_move = None
        self.winner = None
        self.neighbor_pattern = self.generate_neighbour_pattern()
        self.left_right = DisjointSet(size * size + 2)
        self.top_bottom = DisjointSet(size * size + 2)

        for i in range(size):
            self.top_bottom.union(size * size, i)
            self.top_bottom.union(size * size + 1, size * size - 1 - i)
            self.left_right.union(size * size, i * size)
            self.left_right.union(size * size + 1, i * size + size - 1)

    def generate_neighbour_pattern(self):
        hex_pattern = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
        neighbour_patters = []
        for y in range(self.size):
            for x in range(self.size):
                neighbours = self.neighbours(y, x, hex_pattern)
                neighbour_patters.append([y_ * self.size + x_ for y_, x_ in neighbours])
        return neighbour_patters

    def place_piece(self, move):
        if not self.board[move]:
            self.board[move] = self.current_player
            self.last_move = move
            self.empty_spaces.remove(move)
            if self.check_win(move):
                self.winner = self.current_player
            self.current_player = 1 if self.current_player == 2 else 2
            return True
        return False

    def neighbours(self, y, x, hex_pattern):
        return [(y + y_, x + x_) for y_, x_ in hex_pattern if
                (0 <= (y + y_) < self.size) and (0 <= (x + x_) < self.size)]

    def check_win(self, move):
        player = self.board[move]

        if player == 1:
            for move_ in self.neighbor_pattern[move]:
                if self.board[move_] == player:
                    self.left_right.union(move_, move)
            return self.left_right.connected(self.size * self.size, self.size * self.size + 1)
        elif player == 2:
            for move_ in self.neighbor_pattern[move]:
                if self.board[move_] == player:
                    self.top_bottom.union(move_, move)
            return self.top_bottom.connected(self.size * self.size, self.size * self.size + 1)
        return False

    def clone_board(self):
        return copy.deepcopy(self.board)


    def reset_board(self):
        self.current_player = 1
        self.board = np.zeros(self.size * self.size, dtype=int)
        self.empty_spaces = set(range(self.size * self.size))
        self.last_move = None
        self.winner = None


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
            row_mid += " | ".join(map(color_mapping, self.board[r*self.size:(r+1)*self.size]))
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
