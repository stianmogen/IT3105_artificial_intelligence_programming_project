import random

from torch import nn
import torch


class Anet(nn.Module):
    def __init__(self, board_size, embedding_size=64):
        super(Anet, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(board_size**2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 364),
            nn.ReLU(),
            nn.Linear(364, board_size**2+1)
        )
        self.init_weights()

    def forward(self, x):
        #mask = torch.where(x == 0, 1, 0)[:, 1:]
        output = self.nn(x)

        #masked = torch.mul(output, mask)
        return nn.functional.sigmoid(output)

    def init_weights(self):
        init_range = 0.2
        for layer in self.nn:
            if type(layer) == nn.Linear:
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-init_range, init_range)


if __name__ == "__main__":
    board_size = 7
    embedding_size = 8
    epochs = 100_000
    anet = Anet(board_size=board_size, embedding_size=8)

    batch_size = 128

    x_train = []
    y_train = []
    for _ in range(batch_size):
        x_train.append([random.randint(1, 2)] + [random.randint(0, 2) for _ in range(board_size**2)])
        y_train.append([random.random() for _ in range(board_size**2 + 1)])

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(anet.parameters(), lr=0.001, betas=(0.5, 0.999))
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    anet.train()

    for epoch in range(0, epochs + 1):
        optimizer.zero_grad()
        y_pred = anet(x_train)

        loss = criterion(y_pred, y_train)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss}")
            #print(scheduler.get_last_lr())
        loss.backward()
        optimizer.step()
        #scheduler.step()