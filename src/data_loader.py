"""Data loading, preprocessing, scaling, and stratified splitting for Iris dataset."""

import os
import json
import numpy as np
import pandas as pd

CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"]
SHORT_CLASSES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "Sepal Length (cm)",
    "Sepal Width (cm)",
    "Petal Length (cm)",
    "Petal Width (cm)"
]
FEATURE_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class IrisDataLoader:
    """Handles loading, splitting, and standardizing the Iris Flower dataset."""

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "iris.csv")
        self.data_path = data_path
        self.mean_ = None
        self.std_ = None

    def load_raw_data(self):
        """Load raw pandas DataFrame."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Iris dataset not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Loads data, performs stratified train/test split, standardizes features,
        and generates one-hot encoded targets.
        """
        df = self.load_raw_data()
        X_raw = df[FEATURE_KEYS].values.astype(np.float64)
        
        # Map species names to integer indices 0, 1, 2
        species_series = df["species"].str.lower().str.strip()
        y_int = np.zeros(len(species_series), dtype=np.int64)
        for idx, short_name in enumerate(SHORT_CLASSES):
            y_int[species_series == short_name] = idx

        # Stratified train-test split to ensure equal representation per class
        rng = np.random.default_rng(random_state)
        train_indices = []
        test_indices = []

        for class_idx in range(len(SHORT_CLASSES)):
            class_mask = np.where(y_int == class_idx)[0]
            rng.shuffle(class_mask)
            n_test = int(np.round(len(class_mask) * test_size))
            test_indices.extend(class_mask[:n_test])
            train_indices.extend(class_mask[n_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        X_train_raw = X_raw[train_indices]
        y_train_int = y_int[train_indices]
        X_test_raw = X_raw[test_indices]
        y_test_int = y_int[test_indices]

        # Calculate mean and standard deviation from training set (StandardScaler)
        self.mean_ = np.mean(X_train_raw, axis=0)
        self.std_ = np.std(X_train_raw, axis=0)
        # Avoid division by zero if std is tiny
        self.std_ = np.where(self.std_ < 1e-7, 1.0, self.std_)

        X_train = (X_train_raw - self.mean_) / self.std_
        X_test = (X_test_raw - self.mean_) / self.std_

        # One-hot encoding
        y_train_one_hot = self.to_one_hot(y_train_int, num_classes=3)
        y_test_one_hot = self.to_one_hot(y_test_int, num_classes=3)

        return {
            "X_train": X_train,
            "y_train": y_train_one_hot,
            "y_train_labels": y_train_int,
            "X_test": X_test,
            "y_test": y_test_one_hot,
            "y_test_labels": y_test_int,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "mean": self.mean_,
            "std": self.std_,
            "raw_df": df
        }

    def transform(self, X_input):
        """Standardize a single sample or batch of samples using learned mean/std."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler parameters not fitted. Call prepare_data first.")
        X_arr = np.array(X_input, dtype=np.float64)
        return (X_arr - self.mean_) / self.std_

    @staticmethod
    def to_one_hot(y_indices: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """Convert integer class array into one-hot encoded matrix."""
        one_hot = np.zeros((len(y_indices), num_classes), dtype=np.float64)
        one_hot[np.arange(len(y_indices)), y_indices] = 1.0
        return one_hot

    def save_scaler(self, filepath: str):
        """Save mean and std to JSON."""
        data = {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "feature_names": FEATURE_NAMES,
            "feature_keys": FEATURE_KEYS,
            "class_names": CLASS_NAMES
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_scaler(self, filepath: str):
        """Load mean and std from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.mean_ = np.array(data["mean"], dtype=np.float64)
        self.std_ = np.array(data["std"], dtype=np.float64)
