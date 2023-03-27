import os

import numpy as np
import torch

from game_state import HexGameState
from mcts_agent import MCTSAgent
from nn.anet import Anet2
from nn.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(size, num_games, batch_size, epochs, epsilon, sigma, epsilon_decay, save_interval, time_budget, exploration, embedding_size, buffer_size):
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    actor = Anet2(input_dim=(size ** 2 + 1), output_dim=(size ** 2))
    replayBuffer = ReplayBuffer(capacity=buffer_size)

    for ga in range(1, num_games + 1):
        hex_game = HexGameState(size)
        player1 = MCTSAgent(hex_game, actor=actor, epsilon=epsilon, sigma=sigma, time_budget=time_budget, exploration=exploration)

        print(f"GAME {ga}, epsilon = {epsilon}")
        #hex_game.print_board()
        while hex_game.winner == 0:
            move, visit_dist, q = player1.get_move()
            state = np.append(hex_game.current_player, hex_game.clone_board())
            hex_game.place_piece(move)
            replayBuffer.push((state, visit_dist, q))
            #hex_game.print_board()

        print(f'Player {hex_game.winner} wins!')


        samples = replayBuffer.sample(buffer_size // 2)
        actor.fit(samples)
        print(actor.predict(np.array([[1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])))
        epsilon *= epsilon_decay
        sigma *= epsilon_decay

        if ga % save_interval == 0 or ga == 3:
            actor.save_model(f"{size}X{size}/game{ga}")


if __name__ == "__main__":
    play(size=5,
         num_games=100,
         batch_size=64,
         epochs=1000,
         epsilon=1.5,
         sigma=2,
         epsilon_decay=0.99,
         save_interval=20,
         time_budget=0.2,
         exploration=1,
         embedding_size=3,
         buffer_size=512)
