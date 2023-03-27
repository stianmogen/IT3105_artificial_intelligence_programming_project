import numpy as np


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
            return self.Q / self.N + explore * np.sqrt(np.log(self.parent.N) / self.N)

    def add_children(self, children):
        self.children += children
