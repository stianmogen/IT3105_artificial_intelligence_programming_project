import copy
import random

import matplotlib.pyplot as plt
import numpy as np

from node import Node
from hex.player import PlayerInterface
from utilities import normalize


class MCTSAgent(PlayerInterface):

    def __init__(self, state, actor, epsilon, sigma, exploration, rollouts=500):
        self.rootstate = state
        self.root = Node()
        self.exploration = exploration
        self.rollouts = rollouts
        self.actor = actor
        self.epsilon = epsilon
        self.sigma = sigma

    def get_move(self, plot=False):
        self.search()
        move, visit_distribution = self.best_move(plot)
        return move, visit_distribution, self.root.Q2

    def best_move(self, plot=False):
        # TODO: move plot logic to simulator / ttop class. Just return distribution
        if self.rootstate.winner != 0:
            raise Exception("The board already has a winner")

        size = self.rootstate.size

        """
        visits = np.zeros(size*size)
        
        for child in self.root.children:
            move = child.move
            visits[move] = child.N
        
        visits = self.root.children_N
        """

        visit_distribution = normalize(self.root.children_N)

        if plot:
            self.plot_dist(range(size*size), self.root.children_N)

        # choose the move of the most simulated node breaking ties randomly

        """
        max_value = max(visits)
        max_nodes = [n for n in self.root.children if n.N == max_value]
        best_child = random.choice(max_nodes)
        move = best_child.move
        """


        move = np.argmax(self.root.children_N)

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

    def search(self):
        self.root = Node(move=0)
        self.root.parent = Node()
        self.root.parent.children.append(self.root)


        for _ in range(self.rollouts):
            node, state = self.select_node()
            if random.random() > self.sigma:
                reward = self.actor.predict(state.board, state.current_player)
            else:
                turn = state.current_player
                winner = self.roll_out(state)
                reward = 0 if turn == winner else 1
            self.backup(node, reward)

    def select_node(self):
        node = self.root
        state = copy.deepcopy(self.rootstate)

        while len(node.children) != 0:

            max_value = max(node.children, key=lambda edge: edge.value(self.exploration)).value(self.exploration)

            max_nodes = [n for n in node.children if n.value(self.exploration) == max_value]
            node = random.choice(max_nodes)

            state.place_piece(node.move)
            if node.parent.children_N[node.move] == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(node.children)
            state.place_piece(node.move)

        return node, state

    def expand(self, parent, state):
        children = []
        if state.winner != 0:
            return False  # game is over at this node so nothing to expand

        for move in state.empty_spaces:
            children.append(Node(move, parent))

        parent.add_children(children)
        return True

    def roll_out(self, state):
        while state.winner == 0:
            if random.random() < self.epsilon:
                move = random.choice(tuple(state.empty_spaces))
            else:
                move = self.actor.best_move(state.board, state.current_player)
            state.place_piece(move)
        return state.winner

    def backup(self, node, reward):
        while node is not None:
            node.N += 1
            node.score += reward
            if node.parent is not None:
                node.parent.children_N[node.move] += 1
                node.parent.children_scores[node.move] += reward
            node = node.parent
            reward = 1 - reward
