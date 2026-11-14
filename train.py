"""Train and evaluate the bike rental neural network."""

import torch


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Run one training or validation epoch."""
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for features, targets in loader:
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(features)
            loss = loss_fn(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(predictions.detach() - targets).sum().item()
        total_samples += batch_size

    mean_loss = total_loss / total_samples
    mean_mae = total_absolute_error / total_samples

    return mean_loss, mean_mae


def train_model(model, train_loader, validation_loader, epochs=100, lr=0.001):
    """Train the model and restore its best validation weights."""
    import copy

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)

        validation_loss, validation_mae = run_epoch(model, validation_loader, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.2f} | "
            f"Val Loss: {validation_loss:.2f} | "
            f"Val MAE: {validation_mae:.2f}"
        )

    model.load_state_dict(best_weights)

    return history, best_epoch


if __name__ == "__main__":
    from model import BikeRentalNet
    from prepare import (
        BikeRentalDataset,
        create_loaders,
        prepare_features,
        split_data,
    )

    torch.manual_seed(2026)

    train, validation, test = split_data()

    x_train, x_validation, x_test, _ = prepare_features(train, validation, test)

    train_dataset = BikeRentalDataset(x_train, train["rentals"].to_numpy())
    validation_dataset = BikeRentalDataset(
        x_validation, validation["rentals"].to_numpy()
    )
    test_dataset = BikeRentalDataset(x_test, test["rentals"].to_numpy())

    train_loader, validation_loader, test_loader = create_loaders(
        train_dataset,
        validation_dataset,
        test_dataset,
    )

    model = BikeRentalNet(input_size=x_train.shape[1])

    history, best_epoch = train_model(
        model,
        train_loader,
        validation_loader,
        epochs=20,
        lr=0.001,
    )

    print("Best epoch:", best_epoch)
