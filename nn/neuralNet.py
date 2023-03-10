import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

from hex.game_state import HexGameState
from nn.qNetwork import DQN
from nn.replayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet:
    def __init__(self,
                 board: HexGameState,
                 batch_size = 128,
                 learning_rate = 0.001,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.observations = board.size
        self.n_observations = 1
        self.policy_net = DQN(self.n_observations, len(board.empty_spaces)).to(device)
        self.target_net = DQN(self.n_observations, len(board.empty_spaces)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayBuffer()


    def train(self):
        return 0

    def predict(self, state):
        return 0

    def best_action(self, state):
        return 0

if __name__ == '__main__':
    net = NeuralNet(HexGameState(3))
    print(net)
    print(device)