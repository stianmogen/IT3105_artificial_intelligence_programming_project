import math
import random
import time
from queue import Queue
from sys import stderr

import numpy as np

from game_state import HexGameState
import copy


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


class MCTSAgent:

    def __init__(self, state: HexGameState, exploration=1):
        self.rootstate = state
        self.root = Node()
        self.exploration = exploration

    def best_move(self):
        print(self.root.children)
        if self.rootstate.winner is not None:
            return None

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children, key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children if n.N == max_value]
        bestchild = random.choice(max_nodes)
        y, x = bestchild.move
        return y, x

    def search(self, time_budget):

        self.root = Node()

        startTime = time.perf_counter()
        num_rollouts = 0
        # do until we exceed our time budget
        while time.perf_counter() - startTime < time_budget:

            start_select_node = time.perf_counter()
            node, state = self.select_node()
            duration_select_node = time.perf_counter() - start_select_node

            turn = state.current_player
            outcome = self.roll_out(state)

            self.backup(node, turn, outcome)

            num_rollouts += 1

            #print(f"select_node: {duration_select_node:.5f} s")

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
