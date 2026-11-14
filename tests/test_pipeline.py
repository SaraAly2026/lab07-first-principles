
"""Tests for the bike rental data pipeline."""

import torch

from model import BikeRentalNet
from prepare import BikeRentalDataset, prepare_features, split_data


def test_data_split():
    """Check that the data split has the expected sizes."""
    train, validation, test = split_data()

    assert len(train) == 3600
    assert len(validation) == 1200
    assert len(test) == 1200


def test_dataset_shapes():
    """Check dataset feature and target shapes."""
    train, validation, test = split_data()

    x_train, _, _, _ = prepare_features(
        train, validation, test
    )

    dataset = BikeRentalDataset(
        x_train,
        train["rentals"].to_numpy(),
    )

    features, target = dataset[0]

    assert features.shape == torch.Size([8])
    assert target.shape == torch.Size([])
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


def test_model_output_shape():
    """Check that the neural network predicts one value per row."""
    model = BikeRentalNet(input_size=8)

    inputs = torch.randn(32, 8)
    predictions = model(inputs)

    assert predictions.shape == torch.Size([32])