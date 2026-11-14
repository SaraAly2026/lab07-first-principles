"""Prepare bike rental data for model training."""

import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import make_rentals


def split_data(seed=2026):
    """Split data into 60% train, 20% validation, 20% test."""
    data = make_rentals()

    train, remaining = train_test_split(
        data,
        test_size=0.4,
        random_state=seed,
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=seed,
    )

    return train, validation, test


def prepare_features(train, validation, test):
    """Fit preprocessing on training data only."""
    numeric_features = [
        "hour",
        "temperature_c",
        "rain_mm",
        "wind_kph",
        "is_weekend",
    ]
    categorical_features = ["station_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )

    x_train = preprocessor.fit_transform(train.drop(columns=["rentals"]))
    x_validation = preprocessor.transform(validation.drop(columns=["rentals"]))
    x_test = preprocessor.transform(test.drop(columns=["rentals"]))

    return x_train, x_validation, x_test, preprocessor


class BikeRentalDataset(Dataset):
    """Store features and targets as float32 tensors."""

    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def create_loaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=32,
    seed=2026,
):
    """Create reproducible DataLoaders."""
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    train, validation, test = split_data()

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Test rows:", len(test))

    x_train, x_validation, x_test, preprocessor = prepare_features(
        train, validation, test
    )

    print("Training features:", x_train.shape)
    print("Validation features:", x_validation.shape)
    print("Test features:", x_test.shape)

    train_dataset = BikeRentalDataset(
        x_train,
        train["rentals"].to_numpy(),
    )

    validation_dataset = BikeRentalDataset(
        x_validation,
        validation["rentals"].to_numpy(),
    )

    test_dataset = BikeRentalDataset(
        x_test,
        test["rentals"].to_numpy(),
    )

    features, target = train_dataset[0]

    print("Dataset length:", len(train_dataset))
    print("Feature shape:", features.shape)
    print("Target shape:", target.shape)
    print("Feature dtype:", features.dtype)
    print("Target dtype:", target.dtype)

    train_loader, validation_loader, test_loader = create_loaders(
        train_dataset,
        validation_dataset,
        test_dataset,
    )

    batch_features, batch_targets = next(iter(train_loader))

    print("Number of training batches:", len(train_loader))
    print("Batch features shape:", batch_features.shape)
    print("Batch targets shape:", batch_targets.shape)
