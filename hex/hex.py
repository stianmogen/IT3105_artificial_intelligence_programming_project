import torch
from torch import optim, nn

from game_state import HexGameState
from manager import MCTSAgent
from nn.qNetwork import DQN
from player import Player
from nn.replayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(size=7, num_games=20, batch_size=28, epochs=10):
    actor = DQN(size ** 2)
    replayBuffer = ReplayBuffer()

    epsilon = 1
    epsilon_decay = 0.98
    for i in range(num_games):
        board = HexGameState(size)
        player1 = MCTSAgent(board, actor=actor, epsilon=epsilon, time_budget=2, exploration=1)
        player2 = MCTSAgent(board, actor=actor, epsilon=epsilon, time_budget=2, exploration=1)
        # self.player2 = Player(name="2", board_size=size)
        print(f"GAME {i}, epsilon = {epsilon}")
        board.print_board()
        while not board.winner:
            y, x, visit_dist = player1.get_move()
            state = board.clone_board()
            while not board.place_piece(y, x):
                print('Place already filled, try again.')
                y, x, visit_dist = player1.get_move()

            replayBuffer.push([state.flatten(), visit_dist])
            board.print_board()

            if not board.winner:
                y, x, visit_dist = player2.get_move()
                state = board.clone_board()
                while not board.place_piece(y, x):
                    print('Place already filled, try again.')
                    y, x, visit_dist = player2.get_move()

                board.print_board()
                replayBuffer.push([state.flatten(), visit_dist])
        print(f'Player {board.winner} wins!')

        sample = replayBuffer.sample(batch_size)
        x_train = torch.tensor(sample[:][0], dtype=torch.float32)
        y_train = torch.tensor(sample[:][1], dtype=torch.float32)

        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(actor.parameters(), lr=0.0001, betas=(0.5, 0.999))
        actor.train()

        for i in range(epochs):
            optimizer.zero_grad()
            y_pred = actor(x_train)

            loss = criterion(y_pred, y_train)
            if i == 0 or i == epochs-1:
                print(f"Epoch {i} loss: {loss}")
            loss.backward()
            optimizer.step()
        actor.eval()
        epsilon *= epsilon_decay


if __name__ == "__main__":
    play(size=5, num_games=50, batch_size=128, epochs=200)
