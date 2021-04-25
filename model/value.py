import torch
import torch.nn as nn


class Value(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = None
        self.conv2 = None

    def forward(self, state):
        state = torch.Tensor(state)
        return self.conv1(state)
