import numpy as np

from hex.parameters import Parameters

p = Parameters()

"""
Node class for representing tree structure in Monte Carlo Search 
"""
class Node:
    def __init__(self, move=None, parent=None):
        """
        :param move: the move corresponding to the node
        :param parent: the parent node
        children: the list of children for the node
        N: number of visits for the node
        score: the score of the node
        """
        self.move = move
        self.parent = parent
        self.children = []
        self.N = 0
        self.score = 0

    @property
    def Q(self):
        """
        The Q value represents the average score for a node for each visit
        :return: average score
        """
        return self.score / self.N

    def add_children(self, children):
        """
        adds children to list of children
        :param children: children to be added
        """
        self.children += children

    def value(self, c):
        """
        calculates the nodes value to be used in node selection
        :param c: exploration
        :return: calculated value
        """
        if self.N == 0:
            # if node is not visited, returns 0 if no exploration
            # else returns inf
            return 0 if c == 0 else np.Inf
        else:
            # calculate the value based average score,
            # and on the parent nodes and own visit count and the exploration c
            return self.Q + c * np.sqrt(np.log(self.parent.N) / (1 + self.N))