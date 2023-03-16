import torch
from torch import optim, nn
import os
from game_state import HexGameState
from manager import MCTSAgent
from nn.qNetwork import DQN
from player import Player
from nn.replayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(size, num_games, batch_size, epochs, epsilon, epsilon_decay, save_interval, time_budget, exploration):
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    actor = DQN(size ** 2)
    replayBuffer = ReplayBuffer()

    # 4 for g in actual number of games
    for ga in range(num_games):
        # 4 a initialize gameboard
        # 4 b starting board state are default value at initialization
        board = HexGameState(size)
        # 4 c init monte carlo (maybe limit to one mctsa and consider current player)
        player1 = MCTSAgent(board, actor=actor, epsilon=epsilon, time_budget=time_budget, exploration=exploration)
        player2 = MCTSAgent(board, actor=actor, epsilon=epsilon, time_budget=time_budget, exploration=exploration)
        # self.player2 = Player(name="2", board_size=size)
        print(f"GAME {ga}, epsilon = {epsilon}")
        board.print_board()
        # 4 d while ba (board.winner) not in final state
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

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = actor(x_train)

            loss = criterion(y_pred, y_train)
            if epoch == 0 or epoch == epochs-1:
                print(f"Epoch {epoch} loss: {loss}")
            loss.backward()
            optimizer.step()
        actor.eval()
        epsilon *= epsilon_decay

        # 4f) if ga modulo is == 0:
        if ga % save_interval == 0:
            torch.save(actor.state_dict(), f"{size}X{size}/game{ga}")


if __name__ == "__main__":
    play(size=5,
         num_games=50,
         batch_size=128,
         epochs=200,
         epsilon=1,
         epsilon_decay=0.98,
         save_interval=10,
         time_budget=2,
         exploration=1)
