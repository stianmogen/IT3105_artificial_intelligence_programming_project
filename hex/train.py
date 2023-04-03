import os

import numpy as np
import torch

from game_state import HexGameState
from mcts_agent import MCTSAgent
from nn.anet import Anet2
from nn.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reinforce_winner(winner, visit_dist, move):
    if winner:
        for i in range(len(visit_dist)):
            if i == move:
                visit_dist[i] = 1
            else:
                visit_dist[i] = 0
    return visit_dist

def play(size, num_games, batch_size, epochs, epsilon, sigma, epsilon_decay, save_interval, rollouts, exploration, embedding_size, buffer_size):
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    actor = Anet2(input_dim=(size ** 2 + 1), output_dim=(size ** 2))
    replayBuffer = ReplayBuffer(capacity=buffer_size)

    for ga in range(1, num_games + 1):
        hex_game = HexGameState(size)
        player1 = MCTSAgent(hex_game, actor=actor, epsilon=epsilon, sigma=sigma, rollouts=rollouts, exploration=exploration)

        print(f"GAME {ga}, epsilon = {epsilon}")
        while hex_game.winner == 0:
            move, visit_dist, q = player1.get_move(True if (ga % save_interval == 0 or ga == 3) else False)
            state = np.append(hex_game.current_player, hex_game.clone_board())
            hex_game.place_piece(move)
            visit_dist = reinforce_winner(hex_game.winner, visit_dist, move)
            replayBuffer.push((state, visit_dist, q))
            if ga % save_interval == 0 or ga == 3:
                hex_game.print_board()
        print(f'Player {hex_game.winner} wins!')

        samples = replayBuffer.sample(batch_size)
        actor.fit(samples, epochs=epochs)

        epsilon *= epsilon_decay
        sigma *= epsilon_decay

        if ga % save_interval == 0 or ga == 3:
            actor.save_model(f"{size}X{size}/game{ga}")
    actor.save_model(f"{size}X{size}/game{ga}")

if __name__ == "__main__":
    play(size=5,
         num_games=1,
         batch_size=256,
         epochs=1,
         epsilon=10,
         sigma=2,
         epsilon_decay=0.99,
         save_interval=30,
         rollouts=1000,
         exploration=1,
         embedding_size=3,
         buffer_size=1024)
