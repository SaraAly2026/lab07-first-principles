"""Train and evaluate the bike rental neural network."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


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


def evaluate_baselines(
    x_train,
    y_train,
    x_validation,
    y_validation,
):
    """Compare mean prediction with linear regression."""

    mean_prediction = np.full(
        len(y_validation),
        np.mean(y_train),
    )

    mean_mae = mean_absolute_error(
        y_validation,
        mean_prediction,
    )

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    linear_predictions = linear_model.predict(x_validation)

    linear_mae = mean_absolute_error(
        y_validation,
        linear_predictions,
    )

    return mean_mae, linear_mae


def plot_learning_curves(
    history,
    best_epoch,
    filename="learning_curves.png",
):
    """Save training and validation loss curves."""
    from pathlib import Path

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    validation_losses = [row["validation_loss"] for row in history]

    plt.figure(figsize=(9, 5))

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")

    plt.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        label=f"Best Epoch: {best_epoch}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Neural Network Learning Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def run_experiment(
    name,
    train_dataset,
    validation_dataset,
    lr,
    epochs=20,
):
    """Run an experiment with reproducible initialization and batches."""
    from torch.utils.data import DataLoader

    from model import BikeRentalNet

    torch.manual_seed(2026)

    generator = torch.Generator().manual_seed(2026)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = BikeRentalNet(input_size=8, hidden_size=64)

    history, best_epoch = train_model(
        model,
        train_loader,
        validation_loader,
        epochs=epochs,
        lr=lr,
    )

    best_row = history[best_epoch - 1]

    return (
        {
            "experiment": name,
            "learning_rate": lr,
            "best_epoch": best_epoch,
            "best_validation_mae": best_row["validation_mae"],
            "final_train_loss": history[-1]["train_loss"],
            "final_validation_loss": history[-1]["validation_loss"],
        },
        history,
        best_epoch,
    )


def evaluate_final_model(model, test_loader):
    """Evaluate the selected model on untouched test data."""
    loss_fn = torch.nn.MSELoss()

    test_loss, test_mae = run_epoch(
        model,
        test_loader,
        loss_fn,
    )

    return {
        "test_mse": test_loss,
        "test_mae": test_mae,
    }


def train_selected_model(train_dataset, validation_dataset):
    """Train the selected configuration with a reproducible seed."""
    from torch.utils.data import DataLoader

    from model import BikeRentalNet

    torch.manual_seed(2026)

    generator = torch.Generator().manual_seed(2026)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = BikeRentalNet(input_size=8, hidden_size=64)

    history, best_epoch = train_model(
        model,
        train_loader,
        validation_loader,
        epochs=20,
        lr=0.01,
    )

    return model, best_epoch


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
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv("reports/history.csv", index=False)
    print("Best epoch:", best_epoch)

    mean_mae, linear_mae = evaluate_baselines(
        x_train,
        train["rentals"].to_numpy(),
        x_validation,
        validation["rentals"].to_numpy(),
    )

    print(f"Mean Baseline Validation MAE: {mean_mae:.2f}")
    print(f"Linear Regression Validation MAE: {linear_mae:.2f}")
    neural_mae = history[best_epoch - 1]["validation_mae"]
    print(f"Neural Network Validation MAE: {neural_mae:.2f}")

    plot_learning_curves(history, best_epoch)
    print("Learning curves saved to reports/learning_curves.png")

    experiments = [
        ("Main Run", 0.001),
        ("Low Learning Rate", 0.0001),
        ("High Learning Rate", 0.01),
    ]

    experiment_results = []

    for name, learning_rate in experiments:
        print(f"\nRunning experiment: {name}")

        result, exp_history, exp_best_epoch = run_experiment(
            name,
            train_dataset,
            validation_dataset,
            lr=learning_rate,
            epochs=20,
        )

        experiment_results.append(result)
        filename = name.lower().replace(" ", "_") + "_curves.png"

        plot_learning_curves(
            exp_history,
            exp_best_epoch,
            filename=filename,
        )

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(output_dir / "experiments.csv", index=False)

    print("\nExperiment Results:")
    print(results_df.to_string(index=False))
    print("Experiments saved to reports/experiments.csv")

    import json

    print("\nTraining selected final model...")

    final_model, final_best_epoch = train_selected_model(
        train_dataset,
        validation_dataset,
    )

    final_metrics = evaluate_final_model(
        final_model,
        test_loader,
    )

    final_metrics["selected_learning_rate"] = 0.01
    final_metrics["best_epoch"] = final_best_epoch
    final_metrics["hidden_size"] = 64

    with open("reports/metrics.json", "w") as file:
        json.dump(final_metrics, file, indent=4)

    print("\nFinal Test Results:")
    print(f"Test MSE: {final_metrics['test_mse']:.2f}")
    print(f"Test MAE: {final_metrics['test_mae']:.2f}")
    print("Metrics saved to reports/metrics.json")
