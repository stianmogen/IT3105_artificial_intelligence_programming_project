import os

import numpy as np
import torch

from agents.mcts_agent import MCTSAgent
from game_state import HexGameState
from nn.anet import Anet2
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(size, num_games, batch_size, epochs, epsilon, sigma, epsilon_decay, sigma_decay, save_interval, rollouts, exploration, buffer_size):
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    actor = Anet2(board_size=(size))
    replayBuffer = ReplayBuffer(capacity=buffer_size)

    for ga in range(1, num_games + 1):
        hex_game = HexGameState(size)
        player1 = MCTSAgent(hex_game, actor=actor, epsilon=epsilon, sigma=sigma, rollouts=rollouts, exploration=exploration)

        print(f"GAME {ga}, epsilon = {epsilon}, sigma = {sigma}")
        while hex_game.winner == 0:
            move, visit_dist, q = player1.get_move(True if (ga % save_interval == 0 or ga == 3) else False)
            board = hex_game.clone_board()
            if hex_game.current_player == 2:
                board = np.array([[1 if tile == 2 else (2 if tile == 1 else 0)]for tile in board])
                board = board.reshape((size, size)).T.flatten()
                visit_dist = visit_dist.reshape((size, size)).T.flatten()

            replayBuffer.push((board, visit_dist, q))

            hex_game.place_piece(move)
            if ga % save_interval == 0 or ga == 3:
                hex_game.print_board()
        print(f'Player {hex_game.winner} wins!')
        samples = replayBuffer.sample(batch_size)
        actor.fit(samples, epochs=epochs)

        epsilon *= epsilon_decay
        sigma *= sigma_decay

        if ga % save_interval == 0 or ga == 3:
            actor.save_model(f"{size}X{size}/game{ga}")
    actor.save_model(f"{size}X{size}/game{ga}")


if __name__ == "__main__":
    play(size=7,
         num_games=1000,
         batch_size=64,
         epochs=5,
         epsilon=1.00,
         sigma=2,
         epsilon_decay=0.998,
         sigma_decay=0.999,
         save_interval=30,
         rollouts=800,
         exploration=1,
         buffer_size=512)
