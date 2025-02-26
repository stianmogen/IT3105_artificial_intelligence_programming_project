import os
import random

import numpy as np
from keras.models import load_model

from hex.parameters import Parameters
from nn.anet import Anet2
from train import HexGameState
from train import MCTSAgent
import matplotlib.pyplot as plt

p = Parameters()

"""
Actor class to keep track of model name, and model object
"""
class Actor:
    def __init__(self, name, model):
        self.name = name
        self.model = model

def plot_results(results):
    # extract player names
    players = list(set([result[0] for game in results for result in game]))

    # create subplots for each game
    fig, axs = plt.subplots(nrows=len(results), figsize=(8, 6), sharex=True)
    if len(results) == 1:
        axs = [axs]

    # iterate over games
    for i, game in enumerate(results):

        # extract scores for this game
        scores = np.zeros(len(players))
        for result in game:
            scores[players.index(result[0])] = result[1]

        # plot stacked bar chart
        axs[i].bar(players, scores)
        axs[i].set_title(f"Game {i + 1}")
        axs[i].set_ylim(0, p.t_num_games)  # Set y-axis limit for consistency

    # add axis labels and title
    fig.suptitle("Player Scores for Multiple Games")

    # adjust spacing and display chart
    fig.tight_layout()
    plt.show()


def run_series(actor1, actor2, board_size, num_games, epsilon, rollouts, sigma, alpha, exploration):
    """
    :param actor1: actor for player 1
    :param actor2: actor for player 2
    :param board_size: num of rows and columns
    :param num_games: the amount of games to be played in one series
    :param epsilon:
    :param rollouts: num of rollouts
    :param sigma:
    :param exploration:
    :return: winning statistics for the players
    """
    actors = [actor1, actor2]
    board = HexGameState(board_size)

    if p.against_mcts:
        # if we are to play against pure mcts, we set epsilon and sigma to 1 and change display name to MCTS
        agent1 = MCTSAgent(board, actor=actor1.model, epsilon=1, sigma=1, alpha=alpha, rollouts=rollouts, exploration=exploration)
        actor1.name = "MCTS"
    else:
        agent1 = MCTSAgent(board, actor=actor1.model, epsilon=epsilon, sigma=sigma, alpha=alpha, rollouts=rollouts, exploration=exploration)

    agent2 = MCTSAgent(board, actor=actor2.model, epsilon=epsilon, sigma=sigma, alpha=alpha, rollouts=rollouts, exploration=exploration)
    agents = [agent1, agent2]
    w = random.randint(0, 1)  # Starting actor index either 0 or 1
    b = 1 - w  # Second actor index is the one not starting

    # Actor1 wins - actor2 wins
    statistics = [0, 0]

    for i in range(num_games):
        player1 = agents[w]
        player2 = agents[b]
        while not board.winner:
            move, _, _ = player1.get_move()
            player1.move(move)
            player2.move(move)
            board.place_piece(move)
            board.print_board() if p.visualize_board else None
            if not board.winner:
                move, _, _ = player2.get_move()
                player1.move(move)
                player2.move(move)
                board.place_piece(move)
                board.print_board() if p.visualize_board else None
        board.print_board() if p.visualize_board else None

        winner = w if board.winner == 1 else b
        statistics[winner] += 1
        print(f"Winner is {actors[winner].name}") if p.print_winner else None

        w = (w + 1) % 2  # Swapping actor starting
        b = 1 - w

        board.reset_board()
        player1.reset_root()
        player2.reset_root()

    return statistics


def run_tournament():
    actors = []
    size = p.board_size
    actors_dir = f"{size}X{size}"
    results = []
    for filename in os.listdir(actors_dir):
        f = os.path.join(actors_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if filename in p.model_files:
                model = Anet2(board_size=size, load_path=f)
                actors.append(Actor(filename, model))
    for i in range(len(actors)):
        for j in range(i+1, len(actors)):
            print(f"{actors[i].name} vs {actors[j].name}")
            actor1_wins, actor2_wins = run_series(actors[i], actors[j], board_size=size, num_games=p.t_num_games, epsilon=p.t_epsilon, sigma=p.t_sigma, alpha=p.t_alpha, rollouts=p.t_rollouts,
                                    exploration=p.t_exploration)
            print(f"{actors[i].name} victories: {actor1_wins}")
            print(f"{actors[j].name} victories: {actor2_wins}")
            results.append([[actors[i].name, actor1_wins], [actors[j].name, actor2_wins]])
    if results:
        plot_results(results)


def load_saved_model(filename):
    model = load_model(filename, compile=False)
    print(f"Model {filename} loaded succesfully")
    return model


for i in range(p.t_num_tournaments):
    run_tournament()
