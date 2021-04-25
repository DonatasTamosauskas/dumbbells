import torch
import torch.nn as nn


class FcSimpleDqn:
    def __init__(self, input_shape, outputs, hidden=64):
        torch.manual_seed(1423)
        # super().__init__()
        # self.hidden = nn.Linear(input_shape, hidden)
        # self.head = nn.Linear(hidden, outputs)
        linear1 = nn.Linear(input_shape, hidden)
        linear2 = nn.Linear(hidden, outputs)
        activation = nn.ReLU()
        self.model = nn.Sequential(linear1, activation, linear2, activation)

    #
    # def forward(self, x):
    #     x = F.relu(self.hidden(x))
    #     return self.head(x)
