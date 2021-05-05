import torch.nn as nn
import torch.nn.functional as F


class FcSimpleDqn(nn.Module):
    def __init__(self, input_shape, outputs, hidden=64):
        super().__init__()
        self.hidden = nn.Linear(input_shape, hidden)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.head(x)


class DeepFcDqn(nn.Module):
    def __init__(self, input_shape, outputs, hidden=64):
        super().__init__()
        self.hidden = nn.Linear(input_shape, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.hidden3 = nn.Linear(hidden, hidden)
        self.hidden4 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        return self.head(x)


class SkipConnFcDqn(nn.Module):
    def __init__(self, input_shape, outputs, hidden=64):
        super().__init__()
        self.hidden = nn.Linear(input_shape, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.hidden3 = nn.Linear(hidden, hidden)
        self.hidden4 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        x1 = F.relu(self.hidden(x))
        x2 = F.relu(self.hidden2(x1))
        x3 = F.relu(self.hidden3(x2 + x1))
        x4 = F.relu(self.hidden4(x3 + x2 + x1))
        return self.head(x4 + x3 + x2 + x1)


class DeepFcNormDqn(nn.Module):
    def __init__(self, input_shape, outputs, hidden=64):
        super().__init__()
        self.hidden = nn.Linear(input_shape, hidden)
        self.norm = nn.BatchNorm1d(hidden)

        self.hidden2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.BatchNorm1d(hidden)

        self.hidden3 = nn.Linear(hidden, hidden)
        self.norm3 = nn.BatchNorm1d(hidden)

        self.hidden4 = nn.Linear(hidden, hidden)
        self.norm4 = nn.BatchNorm1d(hidden)

        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = self.norm(F.relu(self.hidden(x)))
        x = self.norm2(F.relu(self.hidden2(x)))
        x = self.norm3(F.relu(self.hidden3(x)))
        x = self.norm4(F.relu(self.hidden4(x)))
        return self.head(x)


class SkipConnNormFcDqn(nn.Module):
    def __init__(self, input_shape, outputs, hidden=64):
        super().__init__()
        self.hidden = nn.Linear(input_shape, hidden)
        self.norm = nn.BatchNorm1d(hidden)

        self.hidden2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.BatchNorm1d(hidden)

        self.hidden3 = nn.Linear(hidden, hidden)
        self.norm3 = nn.BatchNorm1d(hidden)

        self.hidden4 = nn.Linear(hidden, hidden)
        self.norm4 = nn.BatchNorm1d(hidden)

        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        x1 = self.norm(F.relu(self.hidden(x)))
        x2 = self.norm2(F.relu(self.hidden2(x1)))
        x3 = self.norm3(F.relu(self.hidden3(x2 + x1)))
        x4 = self.norm4(F.relu(self.hidden4(x3 + x2 + x1)))
        return self.head(x4 + x3 + x2 + x1)


class CnnDqn(nn.Module):
    def __init__(self, input_shape, outputs, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 16, kernel_size=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=1)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=1)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        x = F.relu(self.hidden(x))
        return self.head(x)
