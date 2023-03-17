import torch.nn as nn
import torch.nn.functional as F
import torch


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN(nn.Module):
    def __init__(self, size, embedding_size=64):
        super(DQN, self).__init__()
        self.nn = nn.Sequential(
            nn.Embedding(size + 1, embedding_size),
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*(size+1), size)
        )
        self.init_weights()

    def forward(self, x):
        mask = torch.where(x == 0, 1, 0)[:, 1:]
        output = self.nn(x)
        masked = torch.mul(output, mask)
        normalized = torch.div(masked, torch.sum(masked))
        return normalized

    def init_weights(self):
        init_range = 0.2
        for layer in self.nn:
            if type(layer) == nn.Linear:
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-init_range, init_range)
