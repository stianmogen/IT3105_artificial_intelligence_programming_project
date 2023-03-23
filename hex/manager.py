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

    def value(self, explore, is_maximizing):
        if self.N == 0:
            if explore == 0:
                return 0
            else:
                return np.inf
        else:
            if is_maximizing:
                return self.Q / self.N + explore * np.sqrt(2 * np.log(self.parent.N) / self.N)
            else:
                return self.Q / self.N - explore * np.sqrt(2 * np.log(self.parent.N) / self.N)

    def add_children(self, children):
        self.children += children


class MCTSAgent(PlayerInterface):

    def __init__(self, state: HexGameState, actor, epsilon=1, sigma=1.5, exploration=1, time_budget=5):
        self.rootstate = state
        self.root = Node()
        self.exploration = exploration
        self.time_budget = time_budget
        self.board_size = state.size
        self.actor = actor
        self.epsilon = epsilon
        self.sigma = sigma

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

        #self.plot_dist(range(size*size), visits)

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
        visits = np.zeroes(size*size)
        for child in self.root.children:
            move = child.move
            visits[move] = child.N
        D = normalize(visits)
        return self.rootstate.board, D

    def search(self, time_budget):
        '''
        last_move = self.rootstate.last_move
        if self.root.move and last_move != self.root.move:
            for child in self.root.children:
                if child.move == last_move:
                    self.root = child
                    self.root.parent = None
        '''

        self.root = Node()

        startTime = time.perf_counter()
        num_rollouts = 0
        while time.perf_counter() - startTime < time_budget:
            node, state = self.select_node()
            winner = self.roll_out(state)
            self.backup(node, winner)
            num_rollouts += 1
        """
        stderr.write("Ran " + str(num_rollouts) + " rollouts in " + \
                     str(time.perf_counter() - startTime) + " sec\n")
        stderr.write("Node count: " + str(self.tree_size()) + "\n")
        """

    def select_node(self):
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = copy.deepcopy(self.rootstate)
        # stop if we find reach a leaf node

        while len(node.children) != 0:
            # decend to the maximum value node, break ties at random
            if state.current_player == 1:
                max_value = max(node.children, key=lambda n: n.value(self.exploration, True)).value(self.exploration, True)
                nodes = [n for n in node.children if n.value(self.exploration, True) == max_value]
            else:
                min_value = min(node.children, key=lambda n: n.value(self.exploration, False)).value(self.exploration, False)
                nodes = [n for n in node.children if n.value(self.exploration, False) == min_value]

            node = random.choice(nodes)
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
        if random.random() > self.sigma:
            input = np.expand_dims(np.append(state.current_player, state.board), axis=0)
            move = self.actor.best_move(input)
        while state.winner is None:
            move = random.choice(tuple(moves))
            state.place_piece(move)
        return state.winner

    def backup(self, node, winner):
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play

        reward = 1 if winner == 1 else -10

        while node is not None:
            node.N += 1
            node.Q += (reward - node.Q / (1 + node.N))
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
