import torch
from torch import optim, nn

from game_state import HexGameState
from manager import MCTSAgent
from nn.qNetwork import DQN
from player import Player
from nn.replayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HexGame:
    def __init__(self, size):
        self.board = HexGameState(size)
        self.actor = DQN(size**2)
        self.player1 = MCTSAgent(self.board, actor=self.actor, time_budget=1, exploration=1)
        self.player2 = MCTSAgent(self.board, actor=self.actor, time_budget=1, exploration=1)
        self.replayBuffer = ReplayBuffer()

        #self.player2 = Player(name="2", board_size=size)

    def play(self, num_games=20, batch_size=28, epochs=10):
        for i in range(num_games):

            print(f"GAME {i}")
            self.board.print_board()
            while not self.board.winner:
                y, x, visit_dist = self.player1.get_move()
                state = self.board.clone_board()
                while not self.board.place_piece(y, x):
                    print('Place already filled, try again.')
                    y, x, visit_dist = self.player1.get_move()

                self.replayBuffer.push([state.flatten(), visit_dist])
                self.board.print_board()

                if not self.board.winner:
                    y, x, visit_dist = self.player2.get_move()
                    state = self.board.clone_board()
                    while not self.board.place_piece(y, x):
                        print('Place already filled, try again.')
                        y, x, visit_dist = self.player2.get_move()

                    self.board.print_board()
                    self.replayBuffer.push([state.flatten(), visit_dist])
            print(f'Player {self.board.winner} wins!')

            sample = self.replayBuffer.sample(batch_size)
            x_train = torch.tensor(sample[:][0], dtype=torch.float32)
            y_train = torch.tensor(sample[:][1], dtype=torch.float32)

            criterion = nn.SmoothL1Loss()
            optimizer = optim.Adam(self.actor.parameters(), lr=0.01, betas=(0.5, 0.999))
            self.actor.train()

            for i in range(epochs):
                optimizer.zero_grad()
                y_pred = self.actor(x_train)

                loss = criterion(y_pred, y_train)
                print(f"Epoch {i} loss: {loss}")
                loss.backward()
                optimizer.step()
            self.actor.eval()
            self.board.reset()


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
    HexGame(5).play(20, 128, 10)
