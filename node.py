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
        self.N = 0
        self.score = 0

    @property
    def Q(self):
        return self.score / self.N

    def add_children(self, children):
        self.children += children

    def value(self, c):
        if self.N == 0:
            return 0 if c == 0 else np.Inf
        else:
            return self.Q + c * np.sqrt(np.log(self.parent.N) / (1 + self.N))