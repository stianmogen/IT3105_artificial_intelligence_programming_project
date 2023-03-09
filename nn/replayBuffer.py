from collections import namedtuple, deque
import random

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayBuffer:

    def __init__(self, capacity=256):
        # memory is a dq with capacity defined at instantiation else 256
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, case):
        # we push a new case to memory, if max size is reached we pop left
        if len(self.memory) >= self.capacity:
            self.memory = self.memory[1:]
        self.memory.append(case)


    def sample(self, batch_size):
        # sample method might
        return random.sample(self.memory, batch_size)