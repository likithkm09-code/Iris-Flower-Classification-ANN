"""
Artificial Neural Network (ANN) Engine built with pure vectorized NumPy.
Includes Dense layers, ReLU, Softmax, Cross-Entropy Loss, and Adam Optimizer.
"""

import json
import os
import numpy as np


class DenseLayer:
    """Fully Connected (Dense) Neural Network Layer."""

    def __init__(self, in_features: int, out_features: int, init_method: str = "he"):
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method

        # Weight initialization
        if init_method == "he":
            # He / Kaiming normal for ReLU: std = sqrt(2 / in_features)
            std = np.sqrt(2.0 / in_features)
            self.weights = np.random.randn(in_features, out_features) * std
        elif init_method == "xavier":
            # Xavier / Glorot uniform for output: std = sqrt(2 / (in_features + out_features))
            std = np.sqrt(2.0 / (in_features + out_features))
            self.weights = np.random.randn(in_features, out_features) * std
        else:
            self.weights = np.random.randn(in_features, out_features) * 0.01

        self.biases = np.zeros((1, out_features), dtype=np.float64)

        # Gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

        # Cache for backprop
        self.input_cache = None

        # Adam optimizer state
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Z = X * W + b"""
        self.input_cache = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray, l2_reg: float = 0.0) -> np.ndarray:
        """
        Calculates dW, db, and dX.
        dW = X^T * dZ + l2_reg * W
        db = sum(dZ, axis=0)
        dX = dZ * W^T
        """
        N = self.input_cache.shape[0]
        self.dweights = np.dot(self.input_cache.T, dvalues) + (l2_reg * self.weights)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        dinputs = np.dot(dvalues, self.weights.T)
        return dinputs


class ActivationReLU:
    """Rectified Linear Unit (ReLU): f(x) = max(0, x)"""

    def __init__(self):
        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return np.maximum(0.0, x)

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        dinputs = dvalues.copy()
        dinputs[self.input_cache <= 0.0] = 0.0
        return dinputs


class ActivationSoftmax:
    """Softmax Activation: sigma(z)_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))"""

    def __init__(self):
        self.output_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max per row for numerical stability (prevents overflow)
        exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.output_cache = probabilities
        return probabilities


class CategoricalCrossEntropyLoss:
    """Categorical Cross-Entropy Loss with numeric clipping."""

    @staticmethod
    def compute(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Loss = -1/N * sum( y_true * log(clip(y_pred)) )
        """
        y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
        sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return float(np.mean(sample_losses))


class ArtificialNeuralNetwork:
    """
    Modular Artificial Neural Network for Iris Multi-Class Classification.
    Default Architecture:
        Input Layer: 4 Features
        Hidden Layer 1: 16 Neurons (ReLU)
        Hidden Layer 2: 8 Neurons (ReLU)
        Output Layer: 3 Classes (Softmax)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: list = None,
        output_dim: int = 3,
        l2_reg: float = 0.001
    ):
        if hidden_dims is None:
            hidden_dims = [16, 8]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.l2_reg = l2_reg

        # Construct layers
        self.layers = []
        self.activations = []

        # Layer 1
        self.layers.append(DenseLayer(input_dim, hidden_dims[0], init_method="he"))
        self.activations.append(ActivationReLU())

        # Layer 2
        self.layers.append(DenseLayer(hidden_dims[0], hidden_dims[1], init_method="he"))
        self.activations.append(ActivationReLU())

        # Output Layer
        self.layers.append(DenseLayer(hidden_dims[1], output_dim, init_method="xavier"))
        self.softmax = ActivationSoftmax()

        self.loss_fn = CategoricalCrossEntropyLoss()
        self.t = 0  # Adam timestep

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Propagate inputs through all layers and activations."""
        current_input = X
        for layer, activation in zip(self.layers[:-1], self.activations):
            z = layer.forward(current_input)
            current_input = activation.forward(z)

        # Output layer
        logits = self.layers[-1].forward(current_input)
        probabilities = self.softmax.forward(logits)
        return probabilities

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Backpropagate gradients through all layers.
        For Softmax + Cross Entropy, dZ_output = (y_pred - y_true) / N
        """
        N = y_pred.shape[0]
        # Combined gradient of Softmax + Cross-Entropy Loss
        dvalues = (y_pred - y_true) / N

        # Output layer backward
        dvalues = self.layers[-1].backward(dvalues, l2_reg=self.l2_reg)

        # Hidden layers backward in reverse order
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            dvalues = activation.backward(dvalues)
            dvalues = layer.backward(dvalues, l2_reg=self.l2_reg)

    def update_parameters_adam(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """Update network weights and biases using Adam optimizer."""
        self.t += 1
        for layer in self.layers:
            # Update biased 1st moment vector
            layer.m_w = beta1 * layer.m_w + (1.0 - beta1) * layer.dweights
            layer.m_b = beta1 * layer.m_b + (1.0 - beta1) * layer.dbiases

            # Update biased 2nd raw moment vector
            layer.v_w = beta2 * layer.v_w + (1.0 - beta2) * (layer.dweights ** 2)
            layer.v_b = beta2 * layer.v_b + (1.0 - beta2) * (layer.dbiases ** 2)

            # Compute bias-corrected 1st and 2nd moment estimates
            m_w_hat = layer.m_w / (1.0 - (beta1 ** self.t))
            m_b_hat = layer.m_b / (1.0 - (beta1 ** self.t))
            v_w_hat = layer.v_w / (1.0 - (beta2 ** self.t))
            v_b_hat = layer.v_b / (1.0 - (beta2 ** self.t))

            # Update weights and biases
            layer.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            layer.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 250,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        verbose: bool = True
    ):
        """
        Train the neural network using mini-batch gradient descent with Adam.
        Returns training history dictionary.
        """
        num_samples = X_train.shape[0]
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(1, epochs + 1):
            # Shuffle training indices for mini-batching
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred_batch = self.forward(X_batch)

                # Backward pass
                self.backward(y_pred_batch, y_batch)

                # Adam parameter update
                self.update_parameters_adam(learning_rate=learning_rate)

            # Epoch Evaluation
            train_preds = self.forward(X_train)
            train_loss = self.loss_fn.compute(train_preds, y_train)
            train_acc = float(np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1)))

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_preds = self.forward(X_val)
                val_loss = self.loss_fn.compute(val_preds, y_val)
                val_acc = float(np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1)))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            else:
                val_loss, val_acc = None, None

            if verbose and (epoch % 25 == 0 or epoch == 1 or epoch == epochs):
                val_str = f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%" if val_loss is not None else ""
                print(f"Epoch [{epoch:3d}/{epochs:3d}] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}%{val_str}")

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class probabilities."""
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted integer class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def save_model(self, filepath: str, extra_meta: dict = None):
        """Save model architecture, weights, biases, and metadata to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        model_dict = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "l2_reg": self.l2_reg,
            "layers": []
        }

        for i, layer in enumerate(self.layers):
            model_dict["layers"].append({
                "layer_index": i,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist()
            })

        if extra_meta:
            model_dict["metadata"] = extra_meta

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_dict, f, indent=2)

    def load_model(self, filepath: str) -> dict:
        """Load weights, biases, and architecture from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            model_dict = json.load(f)

        self.input_dim = model_dict["input_dim"]
        self.hidden_dims = model_dict["hidden_dims"]
        self.output_dim = model_dict["output_dim"]
        self.l2_reg = model_dict.get("l2_reg", 0.001)

        # Rebuild layers
        self.layers = []
        self.activations = []
        for i, l_data in enumerate(model_dict["layers"]):
            layer = DenseLayer(l_data["in_features"], l_data["out_features"])
            layer.weights = np.array(l_data["weights"], dtype=np.float64)
            layer.biases = np.array(l_data["biases"], dtype=np.float64)
            self.layers.append(layer)
            if i < len(model_dict["layers"]) - 1:
                self.activations.append(ActivationReLU())

        self.softmax = ActivationSoftmax()
        return model_dict.get("metadata", {})
