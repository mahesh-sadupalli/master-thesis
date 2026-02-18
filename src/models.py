"""Neural network model architectures for flow field compression."""

import torch.nn as nn


class RegressionModel(nn.Module):
    """
    Base regression model architecture: 4 -> 64 -> 64 -> 32 -> 4
    """
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.model(x)


class AdvancedRegressionModel(nn.Module):
    """
    Advanced regression model architecture: 4 -> 128 -> 128 -> 64 -> 4
    """
    def __init__(self):
        super(AdvancedRegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)
