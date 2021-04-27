import torch
import torch.nn as nn
import torch.nn.functional as f


class Policy(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, h1_dim, kernel_size=(1, 3))
        self.batch_norm1 = nn.BatchNorm2d(h1_dim)
        self.conv2 = nn.Conv2d(h1_dim, h2_dim, kernel_size=(1, 48))
        self.batch_norm2 = nn.BatchNorm2d(h2_dim)
        self.conv3 = nn.Conv2d(h2_dim + 1,  output_dim, kernel_size=(1, 1))
        self.cash_bias = nn.Parameter(torch.ones(1))
        # TODO: make kernel sizes input to Policy
        # TODO: check how weights are initialized

    def forward(self, state, prev_action):
        # TODO: add normalizations layers?
        # TODO: currently 1 sample at a time - handle multiple samples at once
        state = torch.tensor(state).unsqueeze(0)

        prev_action = torch.tensor(prev_action[1:]).view(1, 1, -1, 1)
        h1 = f.relu(self.conv1.forward(state))
        h2 = f.relu(self.conv2.forward(h1))
        h2_action = torch.cat((h2, prev_action), dim=1)
        h3 = f.relu(self.conv3.forward(h2_action))
        return f.softmax(torch.cat((self.cash_bias, h3.squeeze())), dim=0)
