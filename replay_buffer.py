from collections import namedtuple, deque
import random


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import torch

"""
ReplayBuffer class to store states to be used for training neural network
"""
class ReplayBuffer:
    def __init__(self, capacity=256):
        """
        :param capacity: the buffer size of the memory
        mempory: DQ with capacity = capacity
        """
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, case):
        """
        :param case: the training case to be appended to memory
        """
        # we push a new case to memory, if max size is reached we pop left
        if len(self.memory) >= self.capacity:
            self.memory.popleft()
        self.memory.append(case)

    def sample(self, batch_size):
        """
        fetches a random sample for the memory
        :param batch_size:
        :return: random sample
        """
        if batch_size > self.__len__():
            # if batch_size is greater than the size of memory, we return the entire memory
            batch_size = self.__len__()

        # random sample of memory from batch_size
        res = random.sample(self.memory, batch_size)
        return res

    def __len__(self):
        return len(self.memory)
