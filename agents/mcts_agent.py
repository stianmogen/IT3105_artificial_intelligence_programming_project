import copy
import random

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

    def move(self, move):
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None
                break
        else:
            print("Could not find this part of the search tree")
            self.root = Node()

    def get_move(self):
        self.search()
        move, visit_distribution = self.best_move()
        return move, visit_distribution, self.root.Q

    def best_move(self):
        if self.rootstate.winner != 0:
            raise Exception("The board already has a winner")

        size = self.rootstate.size
        visits = np.zeros(size*size)

        for child in self.root.children:
            move = child.move
            visits[move] = child.N
        visit_distribution = normalize(visits)

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(visits)
        max_nodes = [n for n in self.root.children if n.N == max_value]
        best_child = random.choice(max_nodes)
        move = best_child.move
        return move, visit_distribution

    def search(self):
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
            if node.N == 0:
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
            node = node.parent
            reward = 1 - reward