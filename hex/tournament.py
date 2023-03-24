import os
import random

from hex import HexGameState
from hex import MCTSAgent
from nn.qNetwork import DQN
from nn.anet import Anet2
from keras.models import load_model


def run_series(actor1, actor2, board_size, num_games, epsilon, time_budget, exploration):
    actors = [actor1, actor2]
    w = random.randint(0, 1)  # Starting actor index either 0 or 1
    b = 1 - w  # Second actor index is the one not starting

    # Actor1 wins - actor2 wins
    statistics = [0, 0]

    for i in range(num_games):
        board = HexGameState(board_size)
        player1 = MCTSAgent(board, actor=actors[w], epsilon=epsilon, time_budget=time_budget, exploration=exploration)
        player2 = MCTSAgent(board, actor=actors[b], epsilon=epsilon, time_budget=time_budget, exploration=exploration)
        while not board.winner:
            move, _, _ = player1.get_move()
            board.place_piece(move)
            board.print_board()
            if not board.winner:
                move, _, _ = player2.get_move()
                board.place_piece(move)
                board.print_board()
        print("WINNER", board.winner)
        print("W=", w, "B=", b)
        board.print_board()

        winner = w if board.winner == 1 else b
        statistics[winner] += 1

        print(statistics)
        w = (w + 1) % 2  # Swapping actor starting
        b = 1 - w

    return statistics


def run_tournament():
    actors = []
    actors_names = []
    size = 5
    actors_dir = f"{size}X{size}"
    for filename in os.listdir(actors_dir):
        f = os.path.join(actors_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if filename == "game300.h5" or filename == "game1000.h5":
                print(filename)
                actors_names.append(filename)

                model = Anet2(load_path=f)
                print(f)
                actors.append(model)

    for i in range(len(actors)):
        for j in range(i+1, len(actors)):
            print(f"{actors_names[i]} vs {actors_names[j]}")
            actor1_wins, actor2_wins = run_series(actors[i], actors[j], board_size=size, num_games=100, epsilon=0, time_budget=1,
                                    exploration=1)
            print(f"{actors_names[i]} victories: {actor1_wins}")
            print(f"{actors_names[j]} victories: {actor2_wins}")


def load_saved_model(filename):
    model = load_model(filename, compile=False)
    print(f"Model {filename} loaded succesfully")
    return model


run_tournament()
