import copy
import random

import numpy as np

from node import Node
from hex.player import PlayerInterface
from utilities import normalize

"""
This class implements Monte Carlo Tree Search
Inherits the PlayerInterface class, defining required methods for playing the given game
Uses a tree structure (represented as disjoint set) to keep track of the game states and corresponding rewards.

"""


class MCTSAgent(PlayerInterface):
    def __init__(self, state, actor, epsilon, sigma, exploration, rollouts=500):
        """
        Initializes the MCTSAgent class with the given parameters.

        :param state: initial state of the game
        :param actor: the neural network that predicts the value of a state
        :param epsilon: positive number that decideds the rate of using actor
        :param sigma: positive number that decides the rate of using critic
        :param exploration: the exploration constant
        :param rollouts: the number of rollouts to be performed for each move
        """
        self.rootstate = state
        self.root = Node()
        self.exploration = exploration
        self.rollouts = rollouts
        self.actor = actor
        self.epsilon = epsilon
        self.sigma = sigma

    def move(self, move):
        """
        Updates the root node of the tree to the child node corresponding to the given move.
        If the move is not in the search tree, creates a new root node.

        :param move: the move to be made from the current state
        """
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None
                break
        else:
            # print("Could not find this part of the search tree")
            self.root = Node()

    def get_move(self):
        """
        runs search algorithm to calculate the best move from search algorithm
        :return:
        - move: the best move to play
        - visit_distribution: a visit distribution over all moves based on the visit counts
        - root.Q: the Q value of the root node
        """
        self.search()
        move, visit_distribution = self.best_move()
        return move, visit_distribution, self.root.Q

    def best_move(self):
        """
        the best move is defiened as the node with the greatest number of visits
        :return: chosen move and the visit distribution
        """
        if self.rootstate.winner != 0:
            raise Exception("The board already has a winner")

        # the visits are added to a numpy array, initially each node is visited zero times
        size = self.rootstate.size
        visits = np.zeros(size * size)

        # for each child, add the visits add to visit list at index corresponding to move
        for child in self.root.children:
            move = child.move
            visits[move] = child.N
        # normalized visits for appropriate visit distribution
        visit_distribution = normalize(visits)

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(visits)
        max_nodes = [n for n in self.root.children if n.N == max_value]
        best_child = random.choice(max_nodes)
        move = best_child.move
        return move, visit_distribution

    def search(self):
        """
        For the number of rollouts defined, performs rollout on selected node
        """
        for _ in range(self.rollouts):
            # selects a node with the given state
            node, state = self.select_node()
            if random.random() > self.sigma:
                # if a random value between 0 and 1 is greater than sigma
                # an actor predicts the given reward for the current state given the player
                reward = self.actor.predict(state.board, state.current_player)
            else:
                # else, perform a rollout and compute reward based on the winner
                turn = state.current_player
                winner = self.roll_out(state)
                reward = 0 if turn == winner else 1
            # updates the tree with the reward
            self.backup(node, reward)

    def select_node(self):
        """
        Selects node with greatest calculated value
        Starting search from root node and iterates over children

        :return: The selected node and game state after piece is placed
        """
        node = self.root
        # makes a deepcopy of the rootstate to make changes to during selection process
        state = copy.deepcopy(self.rootstate)
        # while there are children to search
        while len(node.children) != 0:
            # finds the maximum value based on value method for each of the child node
            max_value = max(node.children, key=lambda edge: edge.value(self.exploration)).value(self.exploration)
            # creates a list of all nodes with the corresponding maximum value
            max_nodes = [n for n in node.children if n.value(self.exploration) == max_value]
            # chooses a random node from this list of nodes
            node = random.choice(max_nodes)
            state.place_piece(node.move)
            # if the node is never visisted, it shall not be expanded, and is rather returned with the given state
            if node.N == 0:
                return node, state

        # expands the current node and state
        if self.expand(node, state):
            # returns a random node from the children and places the piece
            node = random.choice(node.children)
            state.place_piece(node.move)

        return node, state

    def expand(self, parent, state):
        """
        Expands the search tree by creating a child for each possible move from current state
        :param parent: parent node to add child node to
        :param state: current state of game
        :return: true or false if there is something to expand
        """
        children = []
        if state.winner != 0:
            return False  # game is over at this node so nothing to expand

        for move in state.empty_spaces:
            # appends each possible move from current state as node with move and parent to children
            children.append(Node(move, parent))
        # add children to parent
        parent.add_children(children)
        return True

    def roll_out(self, state):
        """
        Performs rollout until terminal state is reached
        :param state: the starting state for the rollout
        :return: state.winner
        """
        # While there is not a winner
        while state.winner == 0:
            if random.random() < self.epsilon:
                # Chooses a random move if the random value between 0-1 is smaller than epsilon
                move = random.choice(tuple(state.empty_spaces))
            else:
                # else the best move is choses by action
                move = self.actor.best_move(state.board, state.current_player)
            state.place_piece(move)
        return state.winner

    def backup(self, node, reward):
        """
        Backs up the result of a rollout through the search tree to the root node.
        Updates the values of N and score on the nodes

        :param node: the node to start
        :param reward: the reward gathered in rollout
        """
        while node is not None:
            node.N += 1
            node.score += reward
            node = node.parent
            reward = 1 - reward
