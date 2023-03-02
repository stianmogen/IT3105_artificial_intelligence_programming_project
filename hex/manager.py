import math
import numpy as np
from nn.neuralNet import NeuralNet
from hex import HexBoard

def possible_new_states(state, current_player):
    """
    Going through the board rows
    For each row returning a generator with new possible states after setting 0 element to current player
    """
    for y in range(len(state)):
        for x in range(len(state[0])):
            if state[y][x] == 0:
                yield state[0:y] + [state[y][0:x] + [current_player] + state[y][x + 1:]] + state[y + 1:], y, x


class MiniMaxAgent:

    def __init__(self, game):
        self.game = game

    def minimax(self, state, current_player, is_maximizing, y, x):
        if (score := self.evaluate(state, is_maximizing, y, x)) is not None:
            return score
        return (max if is_maximizing else min)(
            self.minimax(new_state, current_player=1 if current_player == 2 else 2, is_maximizing=not is_maximizing,
                         y=y, x=x)
            for new_state, y, x in possible_new_states(state, current_player=current_player)
        )

    """
    Finding the best move for the current player 
    Generating new possible states from the current situation
    Based on each new state checking minmax of that move
    Changing player to the other because it is the other players turn
    :param state: current game state
    :param current_player: the player that is going to make a move
    :return: positive score with a new state if win is secured or 0 and the last state if not
    """

    def best_move(self, state, current_player):
        for new_state, y, x in possible_new_states(state, current_player):
            score = self.minimax(new_state, 1 if current_player == 2 else 2, is_maximizing=False, y=y, x=x)
            if score > 0:
                break
        return score, new_state, y, x

    def evaluate(self, state, is_maximizing, y, x):
        if self.game.board.check_win(y, x, state):
            return -1 if is_maximizing else 1
        return None


class Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.N = 0
        self.Q = 0
        self.winner = None

    def add_children(self, children):
        self.children += children

    def set_winner(self, winner):
        self.winner = winner

    def get_N(self):
        return self.N

    def get_Q(self):
        return self.Q

class MonteCarlo:
    """
    https://www.youtube.com/watch?v=UXW2yZndl7U
    1. tree traversal UCB1
    2. node expansion
    3. rollout (random simulation)
    4. backpropagation
    """

    def __init__(self, game: HexGame, root: Node, c=1, num_simulations=10):
        self.game = game
        self.root = root
        self.state = {}
        self.action_states = {}
        self.c = c


    """
    Q(s,a) = value (expected final result) of doing action a from state s.
    These appear on the edges between tree nodes.
    Over time (many expansions, rollouts and backpropagations), these Q values approach those found by Minimax.
    """
    def get_q_value(self, action: Node):
        if action == None:
            return self.root.get_Q()
        return action.get_Q()

    """
    with N(s,a) = number of times branch (s,a) has been traversed, and
    P(s,a) = prior probability p(a | s). In AlphaGo, these P(s,a) come from
    the Actor-SL neural network (trained on a database of expert moves).
    """
    def get_n_value(self, state, action=None):
        # n value is depended on action value definition
        if action:
            if action in self.root.children:
                return action.get_N()
        return self.root.get_N()

    """
    random component. should probably move possible_new_states to game class
    potentially 
    """
    def get_random_action(self):
        pass

    def evaluate(self, state):
        # something along these lines
        return NeuralNet.predict(state)

    def rollout(self, state):
        # best action chosen for the rollout, based on predictions from evaluate
        pred = self.evaluate(state)
        return NeuralNet.best_action(pred)

    """
    we will use Upper Confidence Bound (UCT)
    u(s,a) = c * sqrt ( log(N(s)) / ( 1 + N(s,a) ) )
    """
    def exploration_bonus(self, state, action):
        if self.get_n_value(state) == 0:
            return np.inf
        return self.c * np.sqrt(np.log(self.get_n_value(state) / (1 + self.get_n_value(state, action))))

    def choose_branch_max(self, state, action):
        return self.get_q_value() + self.exploration_bonus(state, action)

    def choose_branch_min(self, state, action):
        return self.get_q_value() - self.exploration_bonus(state, action)

    def traverse(self):
        #TODO how tf to traverse
        pass

    def expand(self, state, parent: Node):
        if (False):
            return 0
        children = []
        for move in self.game.check_win(0, 0, state):
            children.appen(Node(move, parent))
        parent.add_children(children)
        return True

