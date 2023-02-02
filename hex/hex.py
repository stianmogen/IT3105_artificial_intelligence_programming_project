class HexBoard:
    def __init__(self, size):
        self.size = size
        # 0 = nobody
        # 1 = player 1
        # 2 = player 2
        self.board = [[0 for x in range(size)]for y in range(size)]
        self.column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def place_piece(self, x, y, player):
        if self.board[x][y] == 0:
            self.board[x][y] = player
            return True
        return False


    def has_winner(self):
        for i in range(self.size):
            if self.check_row_for_winner(i) or self.check_column_for_winner(i):
                return True
        return self.check_diagonal_for_winner()

    '''
    
    '''

    def check_row_for_winner(self, row):
        color = self.board[row][0]

        if color == 0:
            return False
        for i in range(1, self.size):
            if self.board[row][i] != color:
                return False
        return True

    def check_column_for_winner(self, col):
        color = self.board[0][col]
        if color == 0:
            return False
        for i in range(1, self.size):
            if self.board[i][col] != color:
                return False
        return True

    def check_diagonal_for_winner(self):
        color = self.board[0][0]
        if color == 0:
            return False
        for i in range(1, self.size):
            if self.board[i][i] != color:
                return False
        return True

    '''
    def visualize(self):
        for i in range(self.size):
            row = ['-' if j == 0 else j for j in self.board[i]]
            print(' '.join(row))
    '''

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
            else:
                print('Place already filled, try again.')

        self.board.print_board()
        print(f'Player {current_player} wins!')


HexGame(4).play()