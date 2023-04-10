import copy
import random

import numpy as np
import unittest

from disjoint_set import DisjointSet


class HexGameState:
    def __init__(self, size):
        self.size = size
        self.current_player = 1
        self.board = np.zeros(size * size, dtype=int)
        self.empty_spaces = set(range(size * size))
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.last_move = None
        self.neighbour_pattern = self.generate_neighbour_pattern()
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
        if self.board[move] == 0:
            self.board[move] = self.current_player
            self.last_move = move
            self.empty_spaces.remove(move)
            self.update_groups(move)
            self.next_turn()
        else:
            raise Exception("Invalid move, tile already placed here")

    def update_groups(self, move):
        player = self.board[move]
        if player == 1:
            for neighbour in self.neighbour_pattern[move]:
                if self.board[neighbour] == player:
                    self.left_right.union(neighbour, move)
        elif player == 2:
            for neighbour in self.neighbour_pattern[move]:
                if self.board[neighbour] == player:
                    self.top_bottom.union(neighbour, move)
        else:
            raise Exception("This move has an unassigned player")

    def next_turn(self):
        self.current_player = 1 if self.current_player == 2 else 2

    def neighbours(self, y, x, hex_pattern):
        return [(y + y_, x + x_) for y_, x_ in hex_pattern if
                (0 <= (y + y_) < self.size) and (0 <= (x + x_) < self.size)]

    @property
    def winner(self):
        if self.left_right.connected(self.size * self.size, self.size * self.size + 1):
            return 1
        elif self.top_bottom.connected(self.size * self.size, self.size * self.size + 1):
            return 2
        else:
            return 0

    def reset_board(self):
        self.current_player = 1
        self.board = np.zeros(self.size * self.size, dtype=int)
        self.empty_spaces = set(range(self.size * self.size))
        self.last_move = None
        self.left_right = DisjointSet(self.size * self.size + 2)
        self.top_bottom = DisjointSet(self.size * self.size + 2)

        for i in range(self.size):
            self.top_bottom.union(self.size * self.size, i)
            self.top_bottom.union(self.size * self.size + 1, self.size * self.size - 1 - i)
            self.left_right.union(self.size * self.size, i * self.size)
            self.left_right.union(self.size * self.size + 1, i * self.size + self.size - 1)

    def clone_board(self):
        return copy.deepcopy(self.board)

    def print_board(self, board_in=None):
        rows = self.size
        cols = self.size
        if board_in is not None:
            board = board_in
        else:
            board = self.board
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
            row_mid += " | ".join(map(color_mapping, board[r*self.size:(r+1)*self.size]))
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


class Test(unittest.TestCase):
    def test_invalid_input(self):
        hex_game = HexGameState(5)
        hex_game.place_piece(0)
        with self.assertRaises(Exception):
            hex_game.place_piece(0)

    def test_valid_input(self):
        hex_game = HexGameState(5)
        while len(hex_game.empty_spaces) > 0:
            move = random.choice(tuple(hex_game.empty_spaces))
            hex_game.place_piece(move)

    def test_winner_horizontal(self):
        hex_game = HexGameState(5)
        for i in range(5):
            self.assertIs(hex_game.winner, 0)
            hex_game.place_piece(i)  # white
            hex_game.place_piece(5 + i)  # black
        self.assertIs(hex_game.winner, 1)

    def test_winner_vertical(self):
        hex_game = HexGameState(5)
        for i in range(5):
            self.assertIs(hex_game.winner, 0)
            hex_game.place_piece(i * 5)  # white
            hex_game.place_piece(i * 5 + 1)  # black
        self.assertIs(hex_game.winner, 2)

    def test_winner(self):
        hex_game = HexGameState(5)
        sequence = [0, 24, 6, 18, 12, 13, 17, 7, 11, 1, 5, 19, 10, 3, 4, 20, 8]
        for move in sequence:
            self.assertIs(hex_game.winner, 0)
            hex_game.place_piece(move)
        self.assertIs(hex_game.winner, 1)


if __name__ == "__main__":
    unittest.main()