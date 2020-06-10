import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class NN(nn.Module):
    def __init__(self, dim):
        super(NN, self).__init__()
        self.fc1=nn.Linear(dim, 1000)
        self.fc2=nn.Linear(1000, 1000)
        self.fc3=nn.Linear(1000, 1000)
        self.fc4=nn.Linear(1000, 1000)
        self.fc5=nn.Linear(1000, 1)

    def __call__(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.fc3(h)
        h = F.relu(h)
        h = self.fc4(h)
        h = F.relu(h)
        h = self.fc5(h)
        return h

class CNN(nn.Module):
    def __init__(self, dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.view(-1, 16 * 5 * 5)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h
