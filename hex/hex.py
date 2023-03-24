import numpy as np
import torch
from torch import optim, nn
import os
from game_state import HexGameState
from manager import MCTSAgent
from nn.anet import Anet2
from nn.qNetwork import DQN
from nn.testnet import NeuralNet
from player import Player
from nn.replayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(size, num_games, batch_size, epochs, epsilon, sigma, epsilon_decay, save_interval, time_budget, exploration, embedding_size, buffer_size):
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    print(size)
    actor = Anet2(input_dim=(size ** 2 + 1), output_dim=(size ** 2))
    #actor = NeuralNet(nn_dims=(512, 256), board_size=size)
    replayBuffer = ReplayBuffer(capacity=buffer_size)

    # 4 for g in actual number of games
    for ga in range(1, num_games + 1):
        # 4 a initialize gameboard
        # 4 b starting board state are default value at initialization
        board = HexGameState(size)
        # 4 c true init monte carlo (maybe limit to one mctsa and consider current player)
        player1 = MCTSAgent(board, actor=actor, epsilon=epsilon, sigma=sigma, time_budget=time_budget, exploration=exploration)
        #player2 = MCTSAgent(board, actor=actor, epsilon=epsilon, time_budget=time_budget, exploration=exploration)
        player2 = Player(name="2", board_size=size)
        print(f"GAME {ga}, epsilon = {epsilon}")
        # board.print_board()
        # 4 d while ba (board.winner) not in final state
        while not board.winner:

            move, visit_dist, q = player1.get_move(True if ga % save_interval == 0 else False)
            state = np.append(board.current_player, board.clone_board())
            while not board.place_piece(move):
                print('Place already filled, try again.')
                move, visit_dist, q = player1.get_move(True if ga % save_interval == 0 else False)
            replayBuffer.push((state, visit_dist, q))
            if ga % save_interval == 0:
                board.print_board()

            if not board.winner:
                move, visit_dist, q = player1.get_move()
                state = np.append(board.current_player, board.clone_board())
                while not board.place_piece(move):
                    print('Place already filled, try again.')
                    move, visit_dist, q = player1.get_move()

                if ga % save_interval == 0:
                    board.print_board()

                replayBuffer.push([state, visit_dist, q])
        #print(f'Player {board.winner} wins!')

        samples = replayBuffer.sample(buffer_size // 2)
        actor.fit(samples)

        epsilon *= epsilon_decay
        sigma *= epsilon_decay

        # 4f) if ga modulo is == 0:
        if ga % save_interval == 0 or ga == 3:
            actor.save_model(f"{size}X{size}/game{ga}")


if __name__ == "__main__":
    play(size=6,
         num_games=1000,
         batch_size=64,
         epochs=1000,
         epsilon=1.5,
         sigma=2,
         epsilon_decay=0.99,
         save_interval=100,
         time_budget=0.5,
         exploration=1,
         embedding_size=3,
         buffer_size=512)
