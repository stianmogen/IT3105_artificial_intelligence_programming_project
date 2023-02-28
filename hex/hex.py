from manager import MiniMaxAgent


class HexBoard:
    def __init__(self, size):
        self.size = size
        # 0 = nobody
        # 1 = player 1
        # 2 = player 2
        self.board = [[0 for x in range(size)] for y in range(size)]
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def place_piece(self, x, y, player):
        if self.board[x][y] == 0:
            self.board[x][y] = player
            return True
        return False

    def has_winner(self):
        for i in range(self.size):
            if self.check_row_for_winner(i, self.board) or self.check_column_for_winner(i, self.board):
                return True
        return self.check_diagonal_for_winner(self.board)

    def check_win_condition_from_current_state(self, state):
        for i in range(self.size):
            if self.check_row_for_winner(i, state) or self.check_column_for_winner(i, state):
                return True
        return self.check_diagonal_for_winner(state)

    def has_won(self, placed_tile):
        # current_player =
        pass

    """
    Vite om win condition er top-bunn eller venstre høyre
    Vite må vite hva siste trekk ble lagt
    """

    def is_edge(self, i):
        return i == 0 or i == self.size - 1

    def check_winning_state_player_two(self):
        reachable_nodes = []
        for i in range(self.board_size):
            if self.board[i][0] == 2:
                reachable_nodes.append((i, 0))
        for node in reachable_nodes:
            for n in range(-1, 1):
                if (
                        0 <= node[0] + n < self.board_size
                        and self.board[node[0] + n][node[1] + 1] == 2
                ):
                    if node[1] + 1 == self.board_size - 1:
                        return True
                    if (node[0] + n, node[1] + 1) not in reachable_nodes:
                        reachable_nodes.append((node[0] + n, node[1] + 1))
            if (0 <= node[0] - 1 < self.board_size
                    # Check node to the left in case this has not been picked up by earlier search
                    and self.board[node[0] - 1][node[1]] == 2):
                if (node[0] - 1,
                    node[1]) not in reachable_nodes:
                    reachable_nodes.append((node[0] - 1, node[1]))
            if (0 <= node[0] + 1 < self.board_size
                    # Check node to the right in case this has not been picked up by earlier search
                    and self.board[node[0] + 1][node[1]] == 2):
                if (node[0] + 1,
                    node[1]) not in reachable_nodes:
                    reachable_nodes.append((node[0] + 1, node[1]))
        return False

    def reach_right(self, state, player, x, y, visited, rearched=False):
        pass

    def win_top_bot(self, state, x, y, visited, top=False, bot=False):
        pass

    def check_row_for_winner(self, row, board):
        color = board[row][0]

        if color == 0:
            return False
        for i in range(1, self.size):
            if board[row][i] != color:
                return False
        return True

    def check_column_for_winner(self, col, board):
        color = board[0][col]
        if color == 0:
            return False
        for i in range(1, self.size):
            if board[i][col] != color:
                return False
        return True

    def check_diagonal_for_winner(self, board):
        color = board[0][0]
        if color == 0:
            return False
        for i in range(1, self.size):
            if board[i][i] != color:
                return False
        return True

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


class HexGame:
    def __init__(self, size):
        self.board = HexBoard(size)

    def play(self):
        agent = MiniMaxAgent(self)
        current_player = 1
        while not self.board.has_winner():
            self.board.print_board()
            x, y = input(f'Player {current_player} turn. Enter x and y between 0-{self.board.size - 1}: ').split()
            while int(x) > self.board.size - 1 or int(y) > self.board.size - 1:
                x, y = input(f'Player {current_player} turn. Enter x and y between 0-{self.board.size - 1}: ').split()
            x = int(x)
            y = int(y)
            if self.board.place_piece(x, y, current_player):
                current_player = 2 if current_player == 1 else 1
                print(current_player)
                agent.best_move(self.board.board, current_player)

            else:
                print('Place already filled, try again.')

        self.board.print_board()
        print(f'Player {current_player} wins!')


# HexGame(2).play()


def reach_left(y, x, blocked):
    print(y, x)
    if x == 0:
        return True

    blocked[y][x] = True
    if y > 0:
        if not blocked[y - 1][x] and reach_left(y - 1, x, blocked):
            return True
    if x > 0:
        if not blocked[y][x - 1] and reach_left(y, x - 1, blocked):
            return True
    if y < (len(blocked) - 1):
        if not blocked[y + 1][x] and reach_left(y + 1, x, blocked):
            return True
    if x < (len(blocked[0]) - 1):
        if not blocked[y][x + 1] and reach_left(y, x + 1, blocked):
            return True
    if y < (len(blocked) - 1) & x > 0:
        if not blocked[y + 1][x - 1] and reach_left(y + 1, x - 1, blocked):
            return True
    if y > 0 & x < (len(blocked[0]) - 1):
        if not blocked[y - 1][x + 1] and reach_left(y - 1, x + 1, blocked):
            return True

    return False


def reach_right(y, x, blocked):
    print(y, x)

    if x == len(blocked[0]) - 1:
        return True

    blocked[y][x] = True
    if y > 0:
        if not blocked[y - 1][x] and reach_right(y - 1, x, blocked):
            return True
    if x > 0:
        if not blocked[y][x - 1] and reach_right(y, x - 1, blocked):
            return True
    if y < (len(blocked) - 1):
        if not blocked[y + 1][x] and reach_right(y + 1, x, blocked):
            return True
    if x < (len(blocked[0]) - 1):
        if not blocked[y][x + 1] and reach_right(y, x + 1, blocked):
            return True
    if y < (len(blocked) - 1) & x > 0:
        if not blocked[y + 1][x - 1] and reach_right(y + 1, x - 1, blocked):
            return True
    if y > 0 & x < (len(blocked[0]) - 1):
        if not blocked[y - 1][x + 1] and reach_right(y - 1, x + 1, blocked):
            return True

    return False

def reach_top(y, x, blocked):
    print(y, x)
    if y == 0:
        return True

    blocked[y][x] = True
    if y > 0:
        if not blocked[y - 1][x] and reach_top(y - 1, x, blocked):
            return True
    if x > 0:
        if not blocked[y][x - 1] and reach_top(y, x - 1, blocked):
            return True
    if y < (len(blocked) - 1):
        if not blocked[y + 1][x] and reach_top(y + 1, x, blocked):
            return True
    if x < (len(blocked[0]) - 1):
        if not blocked[y][x + 1] and reach_top(y, x + 1, blocked):
            return True
    if y < (len(blocked) - 1) & x > 0:
        if not blocked[y + 1][x - 1] and reach_top(y + 1, x - 1, blocked):
            return True
    if y > 0 & x < (len(blocked[0]) - 1):
        if not blocked[y - 1][x + 1] and reach_top(y - 1, x + 1, blocked):
            return True

    return False


def reach_bottom(y, x, blocked):
    print(y, x)
    if y == len(blocked) - 1:
        return True

    blocked[y][x] = True
    if y > 0:
        if not blocked[y - 1][x] and reach_bottom(y - 1, x, blocked):
            return True
    if x > 0:
        if not blocked[y][x - 1] and reach_bottom(y, x - 1, blocked):
            return True
    if y < (len(blocked) - 1):
        if not blocked[y + 1][x] and reach_bottom(y + 1, x, blocked):
            return True
    if x < (len(blocked[0]) - 1):
        if not blocked[y][x + 1] and reach_bottom(y, x + 1, blocked):
            return True
    if y < (len(blocked) - 1) & x > 0:
        if not blocked[y + 1][x - 1] and reach_bottom(y + 1, x - 1, blocked):
            return True
    if y > 0 & x < (len(blocked[0]) - 1):
        if not blocked[y - 1][x + 1] and reach_bottom(y - 1, x + 1, blocked):
            return True

    return False


if __name__ == "__main__":
    board = [[1, 0, 2, 1],
             [1, 0, 2, 0],
             [2, 0, 2, 0],
             [0, 2, 0, 0]]

    '''
    [[1, 0, 2, 1],
       [1, 0, 2, 0],
          [2, 0, 2, 0],
             [0, 2, 0, 0]]
    '''

    x = 2
    y = 0

    player = board[y][x]
    print(player)

    blocked = [[False if board[i][j] == player else True for j in range(len(board))] for i in range(len(board[0]))]
    print(blocked)
    if player == 1:
        print(reach_left(y, x, blocked))
        print(reach_right(y, x, blocked))
    else:
        print("y", y)
        print(reach_top(y, x, blocked))
        print(reach_bottom(y, x, blocked))
