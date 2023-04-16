import numpy as np

import time
from agents.mcts_agent import MCTSAgent
from game_state import HexGameState
from hex.test_class import TestClass
from nn.anet import Anet2
from replay_buffer import ReplayBuffer
from parameters import Parameters
import os
import tensorflow as tf
import matplotlib.pyplot as plt

p = Parameters()
test_class = TestClass()

start_time = time.time()


def use_gpu_if_present():
    """
    Method to set tensorflow config to gpu if possible
    Cuda will increase performance on tasks that are appropriate for gpus
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    built = tf.test.is_built_with_cuda()
    print("tf is built with CUDA? ", built)
    sys_details = tf.sysconfig.get_build_info()
    print(sys_details)

    gpus = tf.config.list_physical_devices('GPU')

    print("Num GPUs Available: ", len(gpus))
    print(gpus)

    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def reinforce_winner(winner, visit_dist, move):
    """
    Method that generates a distribution if a winner is found
    :param winner: winner
    :param visit_dist: current visit dist
    :param move: winning move
    :return: visit dist
    """
    if winner:
        for i in range(len(visit_dist)):
            if i == move:
                visit_dist[i] = 1
            else:
                visit_dist[i] = 0
    return visit_dist


def play(size, num_games, sample_size, epochs, epsilon, sigma, epsilon_decay, sigma_decay, save_interval, rollouts,
         exploration, buffer_size):
    """
    :param size: size of the game board columns and rows
    :param num_games: number of games in training
    :param sample_size: size of batch to be sent to replay buffer
    :param epochs: epochs for training each game
    :param epsilon: actor ratio
    :param sigma: critic ratio
    :param epsilon_decay: actor use increase per game
    :param sigma_decay: critic use increase per game
    :param save_interval: how often to save the model
    :param rollouts: number of rollouts per move
    :param exploration: exploration bonus
    :param buffer_size: the capacity of the replay buffer
    """
    # create a directory for each board size
    if not os.path.exists(f"{size}X{size}"):
        os.makedirs(f"{size}X{size}")

    # instantiate actor and replay buffer
    actor = Anet2(board_size=(size))
    replayBuffer = ReplayBuffer(capacity=buffer_size)

    # a new hex game state and mcts agent is instantiated for each game
    hex_game = HexGameState(size)
    player1 = MCTSAgent(hex_game, actor=actor, epsilon=epsilon, sigma=sigma, rollouts=rollouts, exploration=exploration)

    for ga in range(0, num_games + 1):
        test_class.get_test_acc(actor, p.board_size) if p.show_test_acc  else None
        print(f"GAME {ga}, epsilon = {player1.epsilon}, sigma = {player1.sigma}")

        while hex_game.winner == 0:
            # gets the move, distribution, and q value from the search
            move, visit_dist, q = player1.get_move()

            if p.plot_dist and ga % save_interval == 0:
                # plots distribution
                plot_dist(size, visit_dist)

            # makes a clone of the game board
            board = hex_game.clone_board()
            if hex_game.current_player == 2:
                # if current player is 2, we must transpose the game board
                # from here we can get the proper visit distribution as if player two is number one
                # this means we can train the model on the same data
                board = np.array([[1 if tile == 2 else (2 if tile == 1 else 0)] for tile in board])
                board = board.reshape((size, size)).T.flatten()
                visit_dist = visit_dist.reshape((size, size)).T.flatten()

            # places the move on the board, and registers it with the agent
            hex_game.place_piece(move)
            player1.move(move)

            # pushes the training data to the replay buffer
            replayBuffer.push((board, visit_dist, 1 - q))

            if p.show_board and ga % save_interval == 0:
                hex_game.print_board()

        print(f'Player {hex_game.winner} wins!')

        hex_game.reset_board()

        # gets a random sample of size = sample_size
        samples = replayBuffer.sample(sample_size)
        # trains the actor with the sample for n epochs
        actor.fit(samples, epochs=epochs)
        # updates epsilon and sigma based on decays
        player1.epsilon -= epsilon_decay
        player1.sigma -= sigma_decay
        if ga % save_interval == 0 or ga == 1:
            actor.save_model(f"{size}X{size}/{p.model_name}-{ga}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time, " seconds")
    actor.save_model(f"{size}X{size}/{p.rollouts}{p.model_name}-{ga}")


if __name__ == "__main__":
    use_gpu_if_present()
    play(size=p.board_size,
         num_games=p.num_games,
         sample_size=p.sample_size,
         epochs=p.epochs,
         epsilon=p.epsilon,
         sigma=p.sigma,
         epsilon_decay=p.epsilon_decay,
         sigma_decay=p.sigma_decay,
         save_interval=p.save_interval,
         rollouts=p.rollouts,
         exploration=p.exploration,
         buffer_size=p.buffer_size)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")


def plot_dist(size, dist):
    plt.bar(size, dist)
    plt.show()
