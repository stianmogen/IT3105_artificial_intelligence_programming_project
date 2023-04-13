import math

import numpy as np

from hex.parameters import Parameters

p = Parameters()

class Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.children_N = np.zeros(p.board_size**2, dtype=np.float32)
        self.children_scores = np.zeros(p.board_size**2, dtype=np.float32)

    @property
    def Q(self):
        return self.parent.children_scores[self.move] / self.parent.children_N[self.move]

    def add_children(self, children):
        self.children += children


    def value(self, c):
        if self.parent.children_N[self.move] == 0:
            return 0 if c == 0 else np.Inf
        else:
            return (self.parent.children_scores[self.move] / self.parent.children_N[self.move]) + c * np.sqrt(np.log(self.parent.parent.children_N[self.parent.move]) / (1 + self.parent.children_N[self.move]))
