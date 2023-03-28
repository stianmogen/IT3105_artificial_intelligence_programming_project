import os

import numpy as np
import torch

from game_state import HexGameState
from mcts_agent import MCTSAgent
from nn.anet import Anet2
from nn.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(size, num_games, batch_size, epochs, epsilon, sigma, epsilon_decay, save_interval, rollouts, exploration, embedding_size, buffer_size):
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    actor = Anet2(input_dim=(size ** 2 + 1), output_dim=(size ** 2))
    replayBuffer = ReplayBuffer(capacity=buffer_size)

    for ga in range(1, num_games + 1):
        hex_game = HexGameState(size)
        player1 = MCTSAgent(hex_game, actor=actor, epsilon=epsilon, sigma=sigma, rollouts=rollouts, exploration=exploration)

        print(f"GAME {ga}, epsilon = {epsilon}")
        #hex_game.print_board()
        while hex_game.winner == 0:
            move, visit_dist, q = player1.get_move()
            state = np.append(hex_game.current_player, hex_game.clone_board())

            hex_game.place_piece(move)
            replayBuffer.push((state, visit_dist, q))
            #hex_game.print_board()

        print(f'Player {hex_game.winner} wins!')

        samples = replayBuffer.sample(batch_size)
        actor.fit(samples, epochs=epochs)
        epsilon *= epsilon_decay
        sigma *= epsilon_decay

        if ga % save_interval == 0 or ga == 3:
            actor.save_model(f"{size}X{size}/game{ga}")


if __name__ == "__main__":
    play(size=4,
         num_games=1000,
         batch_size=128,
         epochs=10,
         epsilon=0.5,
         sigma=2,
         epsilon_decay=1,
         save_interval=100,
         rollouts=200,
         exploration=1,
         embedding_size=3,
         buffer_size=1024)
