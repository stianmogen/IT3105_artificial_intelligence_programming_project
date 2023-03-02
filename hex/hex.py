import copy

from manager import MiniMaxAgent


class HexBoard:
    def __init__(self, size):
        self.size = size
        # 0 = nobody
        # 1 = player 1
        # 2 = player 2
        self.board = [[0 for x in range(size)] for y in range(size)]
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.finished = False

    def place_piece(self, y, x, player):
        if self.board[y][x] == 0:
            self.board[y][x] = player

            if self.check_win(y, x, self.board):
                self.finished = True
            return True
        return False

    def print_board(self):
        rows = len(self.board)
        cols = len(self.board[0])
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

    def reach_left(self, y, x, blocked):
        if x == 0:
            return True

        blocked[y][x] = True
        for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, -1), (-1, 1)]:
            next_row, next_col = y + dr, x + dc
            if (0 <= next_row <= (len(blocked) - 1)) & (0 <= next_col <= (len(blocked[0]) - 1)):
                if not blocked[next_row][next_col] and self.reach_left(next_row, next_col, blocked):
                    return True

        return False

    def reach_right(self, y, x, blocked):
        if x == len(blocked[0]) - 1:
            return True

        blocked[y][x] = True
        for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, -1), (-1, 1)]:
            next_row, next_col = y + dr, x + dc
            if (0 <= next_row <= (len(blocked) - 1)) & (0 <= next_col <= (len(blocked[0]) - 1)):
                if not blocked[next_row][next_col] and self.reach_right(next_row, next_col, blocked):
                    return True

        return False

    def reach_top(self, y, x, blocked):
        if y == 0:
            return True

        blocked[y][x] = True
        for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, -1), (-1, 1)]:
            next_row, next_col = y + dr, x + dc
            if (0 <= next_row <= (len(blocked) - 1)) & (0 <= next_col <= (len(blocked[0]) - 1)):
                if not blocked[next_row][next_col] and self.reach_top(next_row, next_col, blocked):
                    return True

        return False

    def reach_bottom(self, y, x, blocked):
        if y == len(blocked) - 1:
            return True

        blocked[y][x] = True
        for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, -1), (-1, 1)]:
            next_row, next_col = y + dr, x + dc
            if (0 <= next_row <= (len(blocked) - 1)) & (0 <= next_col <= (len(blocked[0]) - 1)):
                if not blocked[next_row][next_col] and self.reach_bottom(next_row, next_col, blocked):
                    return True

        return False

    def check_win(self, y, x, state):
        player = state[y][x]
        blocked = [[False if state[i][j] == player else True for j in range(len(state))] for i in range(len(state[0]))]

        if player == 1:
            return self.reach_left(y, x, copy.deepcopy(blocked)) & self.reach_right(y, x, copy.deepcopy(blocked))
        else:
            return self.reach_top(y, x, copy.deepcopy(blocked)) & self.reach_bottom(y, x, copy.deepcopy(blocked))


class HexGame:
    def __init__(self, size):
        self.board = HexBoard(size)

    def play(self):
        agent = MiniMaxAgent(self)

        self.board.print_board()
        current_player = 1

        _, _, y, x = agent.best_move(self.board.board, current_player)
        self.board.place_piece(y, x, current_player)
        self.board.print_board()
        while not self.board.finished:
            y, x = input(f'Player {current_player} turn. Enter y and x between 0-{self.board.size - 1}: ').split()

            while not (0 <= int(x) <= self.board.size - 1 and 0 <= int(y) <= self.board.size - 1):
                y, x = input(f'Player {current_player} turn. Enter y and x between 0-{self.board.size - 1}: ').split()
            x = int(x)
            y = int(y)

            current_player = 2 if current_player == 1 else 1
            if self.board.place_piece(y, x, current_player):
                self.board.print_board()

                current_player = 2 if current_player == 1 else 1
                _, _, y, x = agent.best_move(self.board.board, current_player)
                print(y, x)
                self.board.place_piece(y, x, current_player)
                print(self.board.board)
                self.board.print_board()
            else:
                print('Place already filled, try again.')

        self.board.print_board()
        print(f'Player {current_player} wins!')


if __name__ == "__main__":
    HexGame(3).play()
