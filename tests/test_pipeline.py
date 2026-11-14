"""Tests for the bike rental data pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BikeRentalNet
from prepare import BikeRentalDataset, prepare_features, split_data
from train import run_epoch


def test_data_split():
    """Check that the data split has the expected sizes."""
    train, validation, test = split_data()

    assert len(train) == 3600
    assert len(validation) == 1200
    assert len(test) == 1200


def test_dataset_shapes():
    """Check dataset feature and target shapes."""
    train, validation, test = split_data()

    x_train, _, _, _ = prepare_features(train, validation, test)

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


def test_scaler_uses_training_data_only():
    """Check that the scaler is fitted on training data only."""
    train, validation, test = split_data()

    _, _, _, preprocessor = prepare_features(train, validation, test)

    scaler = preprocessor.named_transformers_["numeric"]

    numeric_columns = [
        "hour",
        "temperature_c",
        "rain_mm",
        "wind_kph",
        "is_weekend",
    ]

    expected_means = train[numeric_columns].mean().to_numpy()

    assert torch.allclose(
        torch.tensor(scaler.mean_),
        torch.tensor(expected_means),
    )


def test_dataloader_reproducibility():
    """Check that the same seed produces the same batch order."""
    from prepare import create_loaders

    train, validation, test = split_data()
    x_train, x_validation, x_test, _ = prepare_features(train, validation, test)

    train_dataset = BikeRentalDataset(x_train, train["rentals"].to_numpy())
    validation_dataset = BikeRentalDataset(
        x_validation, validation["rentals"].to_numpy()
    )
    test_dataset = BikeRentalDataset(x_test, test["rentals"].to_numpy())

    loader1, _, _ = create_loaders(
        train_dataset, validation_dataset, test_dataset, seed=2026
    )
    loader2, _, _ = create_loaders(
        train_dataset, validation_dataset, test_dataset, seed=2026
    )

    features1, targets1 = next(iter(loader1))
    features2, targets2 = next(iter(loader2))

    assert torch.equal(features1, features2)
    assert torch.equal(targets1, targets2)


def test_optimizer_updates_weights():
    """Check that training updates the model parameters."""

    torch.manual_seed(2026)

    model = BikeRentalNet(input_size=8)

    features = torch.randn(32, 8)
    targets = torch.randn(32)

    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=32)

    original_weights = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    run_epoch(model, loader, loss_fn, optimizer)

    updated_weights = list(model.parameters())

    assert all(
        not torch.equal(before, after)
        for before, after in zip(original_weights, updated_weights, strict=True)
    )


def test_validation_does_not_update_weights():
    """Check that validation leaves model parameters unchanged."""
    torch.manual_seed(2026)

    model = BikeRentalNet(input_size=8)

    features = torch.randn(32, 8)
    targets = torch.randn(32)

    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=32,
    )

    original_weights = [parameter.detach().clone() for parameter in model.parameters()]

    loss_fn = torch.nn.MSELoss()

    run_epoch(model, loader, loss_fn)

    for before, after in zip(original_weights, model.parameters(), strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
def test_model_different_batch_sizes(batch_size):
    """Check model output for different batch sizes."""
    model = BikeRentalNet(input_size=8)

    features = torch.randn(batch_size, 8)
    predictions = model(features)

    assert predictions.shape == torch.Size([batch_size])
