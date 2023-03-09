import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet:
    def __init__(self):
        pass

    def predict(self, state):
        return 0

    def best_action(self, state):
        return 0

if __name__ == '__main__':
    print(device)