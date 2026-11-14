"""Neural network for bike rental prediction."""

import torch
from torch import nn


class BikeRentalNet(nn.Module):
    """Predict bike rentals using a feedforward neural network."""

    def __init__(self, input_size=8, hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


if __name__ == "__main__":
    model = BikeRentalNet()

    sample = torch.randn(32, 8)
    predictions = model(sample)

    print(model)
    print("Input shape:", sample.shape)
    print("Output shape:", predictions.shape)
