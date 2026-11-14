# Lab 07 — Bike Rental Prediction with PyTorch

## 1. Project Overview

In this lab, I built a neural network using PyTorch to predict the number of bike rentals at different stations. The dataset includes information such as the hour, temperature, rainfall, wind speed, weekend status, and station type.

My goal was to understand how a neural network learns, compare its performance with simpler models, and see how changing the learning rate affects the results.

## 2. Data Preparation

I used the provided `bikes_data.py` file to generate 6,000 station-hour observations.

I divided the data into:

- Training: 60% (3,600 rows)
- Validation: 20% (1,200 rows)
- Testing: 20% (1,200 rows)

I used StandardScaler for numerical features and OneHotEncoder for station type. I fitted the preprocessing steps using only the training data to avoid data leakage.

I also created a custom PyTorch Dataset and used DataLoaders with a batch size of 32.

## 3. Neural Network

I created a feedforward neural network using PyTorch with a hidden size of 64.

For training, I used:

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Initial learning rate: 0.001
- Training epochs: 20
- Random seed: 2026

I tracked the training and validation losses and used the best validation epoch to select the model weights.

## 4. Baseline Comparison

I compared my neural network with two simpler prediction methods.

| Model | Validation MAE |
|---|---:|
| Mean Baseline | 9.95 |
| Linear Regression | 9.03 |
| Neural Network (initial) | 5.90 |

The neural network performed better than both baseline models.

## 5. Learning Rate Experiments

I tested three learning rates while keeping the other training settings the same.

| Experiment | Learning Rate | Best Epoch | Validation MAE |
|---|---:|---:|---:|
| Main Run | 0.001 | 20 | 5.90 |
| Low Learning Rate | 0.0001 | 20 | 8.27 |
| High Learning Rate | 0.01 | 16 | 2.62 |

The learning rate of 0.01 gave the best validation performance among the three experiments.

The lower learning rate improved more slowly, while the higher learning rate reached a better result within 20 epochs.

## 6. Learning Curves

The initial learning curves showed that both training and validation loss decreased during training.

The curves stayed close together, which suggests there was no clear evidence of overfitting during the initial 20 epochs.

I saved the learning curves for the main model and each learning rate experiment in the `reports` folder.

## 7. Final Test Results

After selecting the best learning rate using validation data, I evaluated the final model on the test dataset.

| Metric | Result |
|---|---:|
| Test MSE | 14.83 |
| Test MAE | 2.55 |
| Selected Learning Rate | 0.01 |

The test MAE of 2.55 means the model's predictions were off by approximately 2.55 bike rentals on average.

The test result was close to the validation MAE of 2.62, suggesting that the model generalized reasonably well to unseen data.

## 8. Testing

I used pytest to test the data split, dataset shapes, and neural network output shape.

All three automated tests passed.

I also ran a small-batch memorization test to confirm that the neural network could learn a very small dataset.

Results:

- Initial Loss: 33.7721
- Final Loss: 0.0000
- Loss Ratio: 0.000000

## 9. How to Run

Install the required packages:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the training pipeline:

```bash
python train.py
```

Run the automated tests:

```bash
python -m pytest -q
```

Run the small-batch memorization test:

```bash
python check_small_batch.py
```

Check the code quality:

```bash
python -m ruff check .
```

## 10. Results and Reflection

This lab helped me understand the complete process of preparing data, building a neural network, training it, and evaluating its performance.

One important thing I learned was how much the learning rate can affect training. Changing it from 0.001 to 0.01 reduced the validation MAE from 5.90 to 2.62.

I also learned why we should keep training, validation, and test data separate, and why comparing a neural network with simpler baseline models is important.

Overall, the final model performed better than the initial neural network and the baseline models on validation data.