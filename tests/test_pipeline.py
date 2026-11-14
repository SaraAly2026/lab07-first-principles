"""Tests for the bike rental data pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BikeRentalNet
from prepare import BikeRentalDataset, prepare_features, split_data
from train import run_epoch, train_model


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


@pytest.mark.parametrize("hidden_size", [8, 16, 32, 64])
def test_model_different_hidden_sizes(hidden_size):
    """Check that the model works with different hidden layer sizes."""
    model = BikeRentalNet(
        input_size=8,
        hidden_size=hidden_size,
    )

    features = torch.randn(32, 8)
    predictions = model(features)

    assert predictions.shape == torch.Size([32])


def test_data_splits_do_not_overlap():
    """Check that train, validation, and test have no shared rows."""
    train, validation, test = split_data()

    train_indices = set(train.index)
    validation_indices = set(validation.index)
    test_indices = set(test.index)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)


def test_best_weights_restoration():
    """Check that training restores the best validation weights."""
    torch.manual_seed(2026)

    model = BikeRentalNet(input_size=8, hidden_size=16)

    features = torch.randn(32, 8)
    targets = torch.randn(32)

    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=16,
    )

    history, best_epoch = train_model(
        model,
        loader,
        loader,
        epochs=5,
        lr=0.01,
    )

    validation_losses = [epoch["validation_loss"] for epoch in history]

    expected_best_epoch = validation_losses.index(min(validation_losses)) + 1

    assert best_epoch == expected_best_epoch
    assert len(history) == 5


def test_small_batch_memorization():
    """Check that the network can memorize a small batch."""
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

    for _step in range(1000):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(model(features), targets).item()

    assert final_loss < initial_loss * 0.01


def test_restored_model_matches_best_validation_loss():
    """Check that restored weights produce the best validation loss."""
    torch.manual_seed(2026)

    model = BikeRentalNet(input_size=8, hidden_size=16)

    features = torch.randn(32, 8)
    targets = torch.randn(32)

    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=16,
    )

    history, best_epoch = train_model(
        model,
        loader,
        loader,
        epochs=5,
        lr=0.01,
    )

    loss_fn = torch.nn.MSELoss()

    restored_loss, _ = run_epoch(model, loader, loss_fn)

    expected_loss = history[best_epoch - 1]["validation_loss"]

    assert restored_loss == pytest.approx(expected_loss, abs=1e-5)


def test_scaled_features_use_training_statistics():
    """Check training scaling and reuse of training statistics."""
    train, validation, test = split_data()

    x_train, x_validation, _, _ = prepare_features(train, validation, test)

    # First five columns are the scaled numeric features.
    train_means = x_train[:, :5].mean(axis=0)
    validation_means = x_validation[:, :5].mean(axis=0)

    assert torch.allclose(
        torch.as_tensor(train_means, dtype=torch.float32),
        torch.zeros(5),
        atol=1e-5,
    )

    assert not torch.allclose(
        torch.as_tensor(validation_means, dtype=torch.float32),
        torch.zeros(5),
        atol=1e-5,
    )
