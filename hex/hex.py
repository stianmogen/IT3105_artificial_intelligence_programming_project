from game_state import GameState
from manager import MiniMaxAgent


class HexGame:
    def __init__(self, size):
        self.board = GameState(size)

    def play(self):
        agent = MiniMaxAgent(self.board)

        self.board.print_board()

        y, x = agent.best_move()
        self.board.place_piece(y, x)
        self.board.print_board()
        while not self.board.winner:
            y, x = input(f'Player {self.board.current_player} turn. Enter y and x between 0-{self.board.size - 1}: ').split()

            while not (0 <= int(x) <= self.board.size - 1 and 0 <= int(y) <= self.board.size - 1):
                y, x = input(f'Player {self.board.current_player} turn. Enter y and x between 0-{self.board.size - 1}: ').split()
            x = int(x)
            y = int(y)

            if self.board.place_piece(y, x):
                self.board.print_board()

                y, x = agent.best_move()
                self.board.place_piece(y, x)
                self.board.print_board()
            else:
                print('Place already filled, try again.')

        self.board.print_board()
        print(f'Player {self.board.winner} wins!')


if __name__ == "__main__":
    HexGame(3).play()
