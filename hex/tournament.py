import os
import random

import numpy as np
from keras.models import load_model

from nn.anet import Anet2
from train import HexGameState
from train import MCTSAgent


class Actor:
    def __init__(self, name, model):
        self.name = name
        self.model = model


def run_series(actor1, actor2, board_size, num_games, epsilon, rollouts, sigma, exploration):
    actors = [actor1, actor2]
    board = HexGameState(board_size)

    #Pure mcts
    #agent1 = MCTSAgent(board, actor=actor1.model, epsilon=1, sigma=1, rollouts=rollouts, exploration=exploration)

    agent1 = MCTSAgent(board, actor=actor1.model, epsilon=epsilon, sigma=sigma, rollouts=rollouts, exploration=exploration)
    agent2 = MCTSAgent(board, actor=actor2.model, epsilon=epsilon, sigma=sigma, rollouts=rollouts, exploration=exploration)
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
            board.place_piece(move)
            #board.print_board()
            if not board.winner:
                move, _, _ = player2.get_move()
                board.place_piece(move)
                #board.print_board()
        #print("WINNER", board.winner)
        #print("W=", w, "B=", b)
        #board.print_board()

        winner = w if board.winner == 1 else b
        statistics[winner] += 1
        print(f"Winner is {actors[winner].name}")

        w = (w + 1) % 2  # Swapping actor starting
        b = 1 - w

        board.reset_board()

    return statistics


def run_tournament():
    actors = []
    size = 5
    actors_dir = f"{size}X{size}"
    for filename in os.listdir(actors_dir):
        f = os.path.join(actors_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if filename == "game30.h5" or filename == "game480.h5":
                model = Anet2(board_size=size, load_path=f)
                actors.append(Actor(filename, model))

    for i in range(len(actors)):
        for j in range(i+1, len(actors)):
            print(f"{actors[i].name} vs {actors[j].name}")
            actor1_wins, actor2_wins = run_series(actors[i], actors[j], board_size=size, num_games=20, epsilon=0.5, sigma=0.5, rollouts=500,
                                    exploration=1)
            print(f"{actors[i].name} victories: {actor1_wins}")
            print(f"{actors[j].name} victories: {actor2_wins}")


def load_saved_model(filename):
    model = load_model(filename, compile=False)
    print(f"Model {filename} loaded succesfully")
    return model


run_tournament()
