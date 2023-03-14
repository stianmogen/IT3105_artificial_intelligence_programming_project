from game_state import HexGameState
from manager import MCTSAgent
from player import Player


class HexGame:
    def __init__(self, size):
        self.board = HexGameState(size)
        self.player1 = MCTSAgent(self.board, time_budget=5, exploration=1)
        self.player2 = MCTSAgent(self.board, time_budget=5, exploration=1)
        #self.player2 = Player(name="2", board_size=size)

    def play(self):
        self.board.print_board()
        while not self.board.winner:

            y, x = self.player1.get_move()
            while not self.board.place_piece(y, x):
                print('Place already filled, try again.')
                y, x = self.player1.get_move()
            self.board.print_board()

            if not self.board.winner:
                y, x = self.player2.get_move()
                while not self.board.place_piece(y, x):
                    print('Place already filled, try again.')
                    y, x = self.player2.get_move()
                self.board.print_board()

        print(f'Player {self.board.winner} wins!')


    def run(self, epsilon=1, sigma=1):
        """
        TODO Method for running simulations and training
        :param epsilon:
        :param sigma:
        :return:
        """
        while not self.board.winner:
            board_copy = self.board.clone_board()
            for i in range(10):
                reward = self.player1.roll_out_game(board_copy)
                if reward == -10:
                    print(reward)
        print("finished")




if __name__ == "__main__":
    HexGame(7).play()
