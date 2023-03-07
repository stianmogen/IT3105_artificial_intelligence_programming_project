import math
import random
import time
from queue import Queue
from sys import stderr

import numpy as np
from nn.neuralNet import NeuralNet
from game_state import HexGameState, Player
import copy


def possible_new_states(state):
    for y, x in state.empty_spaces:
        new_state = copy.deepcopy(state)
        new_state.place_piece(y, x)
        yield new_state


class MiniMaxAgent:

    def __init__(self, state: HexGameState):
        self.state = state

    def minimax(self, state, is_maximizing):
        if (score := self.evaluate(state)) is not None:
            return score
        return (max if is_maximizing else min)(
            self.minimax(new_state, is_maximizing=not is_maximizing)
            for new_state in possible_new_states(state)
        )

    def best_move(self):
        for y, x in self.state.empty_spaces:
            new_state = copy.deepcopy(self.state)
            new_state.place_piece(y, x)
            score = self.minimax(new_state, is_maximizing=False)
            if score > 0:
                break
        return y, x

    def evaluate(self, state):
        if state.winner is not None:
            if state.winner == -1:
                return 0
            return 1 if self.state.current_player.value == state.winner else -1
        return None


class Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.N = 0
        self.Q = 0
        self.winner = 0

    def value(self, explore):
        if self.N == 0:
            if explore == 0:
                return 0
            else:
                return np.inf
        else:
            return self.Q / self.N + explore * np.sqrt(2 * np.log(self.parent.N) / self.N)

    def add_children(self, children):
        self.children += children

    def set_winner(self, winner):
        self.winner = winner


class MCTSAgent:

    def __init__(self, state: HexGameState, exploration=1):
        self.rootstate = copy.deepcopy(state)
        self.root = Node()
        self.exploration = exploration

    def best_move(self):
        if self.rootstate.winner is not None:
            return None

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children, key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children if n.N == max_value]
        bestchild = random.choice(max_nodes)
        y, x = bestchild.move
        return y, x

    def move(self, move):
        y, x = move
        for child in self.root.children:
            # make the child associated with the move the new root
            if move == child.move:
                child.parent = None
                self.root = child
                self.rootstate.place_piece(y, x)
                return

        # if for whatever reason the move is not in the children of
        # the root just throw out the tree and start over
        self.rootstate.place_piece(y, x)
        self.root = Node()

    def search(self, time_budget):
        """
        Search and update the search tree for a specified amount of time in secounds.
        """
        startTime = time.perf_counter()
        num_rollouts = 0

        # do until we exceed our time budget
        while time.perf_counter() - startTime < time_budget:
            node, state = self.select_node()
            turn = state.current_player
            outcome = self.roll_out(state)
            self.backup(node, turn, outcome)
            num_rollouts += 1

        stderr.write("Ran " + str(num_rollouts) + " rollouts in " + \
                     str(time.perf_counter() - startTime) + " sec\n")
        stderr.write("Node count: " + str(self.tree_size()) + "\n")

    def select_node(self):
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = copy.deepcopy(self.rootstate)

        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # decend to the maximum value node, break ties at random
            max_value = max(node.children, key=lambda n: n.value(self.exploration)).value(self.exploration)
            max_nodes = [n for n in node.children if n.value(self.exploration) == max_value]
            node = random.choice(max_nodes)
            y, x = node.move
            state.place_piece(y, x)

            # if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(node.children)
            y, x = node.move
            state.place_piece(y, x)
        return node, state

    def expand(self, parent, state):
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed HexGameState and add them to the tree.
        """
        children = []
        if state.winner != None:
            # game is over at this node so nothing to expand
            return False

        for move in state.empty_spaces:
            children.append(Node(move, parent))

        parent.add_children(children)
        return True

    def roll_out(self, state: HexGameState):
        moves = state.empty_spaces

        while state.winner == 0:
            y, x = random.choice(tuple(moves))
            state.place_piece(y, x)
            moves.remove((y, x))

        return state.winner

    def backup(self, node, turn, outcome):
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play
        reward = -1 if outcome == turn else 1

        while node != None:
            node.N += 1
            node.Q += reward
            reward = -reward
            node = node.parent

    def set_HexGameState(self, state):
        self.rootstate = copy.deepcopy(state)
        self.root = Node()

    def tree_size(self):
        """
        Count nodes in tree by BFS.
        """
        Q = Queue()
        count = 0
        Q.put(self.root)
        while not Q.empty():
            node = Q.get()
            count += 1
            for child in node.children:
                Q.put(child)
        return count


class MonteCarlo:
    """
    https://www.youtube.com/watch?v=UXW2yZndl7U
    1. tree traversal UCB1
    2. node expansion
    3. rollout (random simulation)
    4. backpropagation
    """

    def __init__(self, game: HexGameState, root: Node, c=1, num_simulations=10):
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
        # TODO how tf to traverse
        pass

    def expand(self, state, parent: Node):
        if self.game.winner:
            return 0
        children = []
        for move in self.game.check_win(0, 0, state):
            children.appen(Node(move, parent))
        parent.add_children(children)
        return True
