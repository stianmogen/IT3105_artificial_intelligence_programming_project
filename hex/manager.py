import random
import time
from queue import Queue
from sys import stderr

import torch

from player import PlayerInterface
import matplotlib.pyplot as plt

import numpy as np

from game_state import HexGameState
from nn.qNetwork import DQN
from nn.testnet import NeuralNet
import copy


def normalize(x):
    return x / x.sum()


class Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.N = 0
        self.Q = 0

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


class MCTSAgent(PlayerInterface):

    def __init__(self, state: HexGameState, actor: NeuralNet, epsilon=1, exploration=1, time_budget=5):
        self.rootstate = state
        self.root = Node()
        self.exploration = exploration
        self.time_budget = time_budget
        self.board_size = state.size
        self.actor = actor
        self.epsilon = epsilon

    def get_move(self):
        self.search(self.time_budget)
        move, visit_distribution = self.best_move()
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None
                break
        return move, visit_distribution, self.root.Q

    def best_move(self):
        if self.rootstate.winner is not None:
            return None

        size = self.rootstate.size
        visits = np.zeros(size*size)

        for child in self.root.children:
            move = child.move
            visits[move] = child.N

        visit_distribution = normalize(visits)

        #self.plot_dist(range(size*size), visit_distribution)

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(visits)
        max_nodes = [n for n in self.root.children if n.N == max_value]
        best_child = random.choice(max_nodes)
        move = best_child.move
        return move, visit_distribution

    def plot_dist(self, size, dist):
        plt.bar(size, dist)
        plt.show()

    def distribution(self):
        if self.rootstate.winner is not None:
            return None

        size = self.rootstate.size
        #output = np.zeros([2, size*size], dtype=object)
        visits = np.zeroes(size*size)
        for child in self.root.children:
            move = child.move
            #output[0][y * size + x] = child.move
            visits[move] = child.N
        #output[1:2, :] = normalize(output[1:2, :])
        D = normalize(visits)
        return self.rootstate.board, D

    def search(self, time_budget):
        last_move = self.rootstate.last_move
        if self.root.move and last_move != self.root.move:
            for child in self.root.children:
                if child.move == last_move:
                    self.root = child
                    self.root.parent = None

        startTime = time.perf_counter()
        num_rollouts = 0
        while time.perf_counter() - startTime < time_budget:
            node, state = self.select_node()
            player = state.current_player
            outcome = self.roll_out(state)

            self.backup(node, player, outcome)
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
            move = node.move
            state.place_piece(move)

            # if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(node.children)
            move = node.move
            state.place_piece(move)
        return node, state

    def expand(self, parent, state):
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed HexGameState and add them to the tree.
        """
        children = []
        if state.winner is not None:
            # game is over at this node so nothing to expand
            return False

        for move in state.empty_spaces:
            children.append(Node(move, parent))

        parent.add_children(children)
        return True

    def roll_out(self, state: HexGameState):
        moves = state.empty_spaces
        while state.winner is None:
            if random.random() < self.epsilon:
                move = random.choice(tuple(moves))
            else:
                input = np.append(state.current_player, state.board)
                preds = self.actor.predict(np.array([input]))
                y, x = self.actor.best_action(preds[0])
                move = y * self.rootstate.size + x

            state.place_piece(move)
        return state.winner

    def backup(self, node, turn, outcome):
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play


        reward = -1 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            reward = -reward
            node = node.parent

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
