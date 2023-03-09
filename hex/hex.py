from game_state import HexGameState
from manager import MCTSAgent
from player import Player


class HexGame:
    def __init__(self, size):
        self.board = HexGameState(size)
        self.player1 = MCTSAgent(self.board)
        self.player2 = Player(name="2", board_size=size)


    def play(self):
        self.board.print_board()
        while not self.board.winner:

            y, x = self.player1.get_move()
            while not self.board.place_piece(y, x):
                print('Place already filled, try again.')
                y, x = self.player1.get_move()
            self.board.print_board()

            y, x = self.player2.get_move()
            while not self.board.place_piece(y, x):
                print('Place already filled, try again.')
                y, x = self.player2.get_move()
            self.board.print_board()

        print(f'Player {self.board.winner} wins!')


if __name__ == "__main__":
    HexGame(4).play()
