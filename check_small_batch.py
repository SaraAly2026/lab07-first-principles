
"""Check whether the network can memorize a small batch."""

import torch

from model import BikeRentalNet
from prepare import (
    BikeRentalDataset,
    prepare_features,
    split_data,
)

torch.manual_seed(2026)

train, validation, test = split_data()
x_train, _, _, _ = prepare_features(train, validation, test)

dataset = BikeRentalDataset(
    x_train,
    train["rentals"].to_numpy(),
)

features = dataset.features[:8]
targets = dataset.targets[:8]

model = BikeRentalNet(input_size=features.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

with torch.no_grad():
    initial_loss = loss_fn(model(features), targets).item()

for step in range(1000):
    optimizer.zero_grad()
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    final_loss = loss_fn(model(features), targets).item()

print(f"Initial Loss: {initial_loss:.4f}")
print(f"Final Loss: {final_loss:.4f}")
print(f"Loss Ratio: {final_loss / initial_loss:.6f}")