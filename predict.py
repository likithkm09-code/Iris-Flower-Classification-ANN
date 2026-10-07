"""
Interactive & Command-Line Prediction Tool for Iris Flower Classification ANN.
Loads the trained model weights and standardization parameters to classify custom flower measurements.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ann_model import ArtificialNeuralNetwork
from src.data_loader import CLASS_NAMES, SHORT_CLASSES, FEATURE_NAMES


class IrisPredictor:
    """Inference engine using saved ANN model weights and scaler statistics."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "saved_models", "iris_ann_model.json")
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run train.py first!")

        self.model = ArtificialNeuralNetwork()
        self.metadata = self.model.load_model(self.model_path)
        self.mean = np.array(self.metadata["mean"], dtype=np.float64)
        self.std = np.array(self.metadata["std"], dtype=np.float64)

    def predict(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> dict:
        """Standardize input features and return predicted class, confidence, and probabilities."""
        raw_features = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=np.float64)
        # Standardize using training mean and std
        scaled_features = (raw_features - self.mean) / self.std
        scaled_input = scaled_features.reshape(1, -1)

        probabilities = self.model.predict_proba(scaled_input)[0]
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        return {
            "predicted_index": predicted_idx,
            "predicted_species": CLASS_NAMES[predicted_idx],
            "short_name": SHORT_CLASSES[predicted_idx],
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 5) for i in range(len(CLASS_NAMES))
            },
            "input_features": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
        }


def print_prediction_result(res: dict):
    """Pretty prints the prediction result to console."""
    print("\n" + "=" * 60)
    print("                ANN INFERENCE RESULT                ")
    print("=" * 60)
    print("Input Measurements:")
    for k, v in res["input_features"].items():
        print(f"  * {k.replace('_', ' ').title():<18}: {v:.2f} cm")

    print("-" * 60)
    print(f"Predicted Species : >>> {res['predicted_species'].upper()} <<<")
    print(f"Prediction Confidence: {res['confidence'] * 100:.2f}%")
    print("-" * 60)
    print("Softmax Probability Distribution:")
    for species, prob in res["probabilities"].items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {species:<16} [{bar}] {prob * 100:6.2f}%")
    print("=" * 60 + "\n")


def run_benchmark_presets(predictor: IrisPredictor):
    """Test 3 canonical samples representing each class."""
    print("\n--- Running Validation on Canonical Flower Samples ---")
    presets = [
        ("Typical Iris setosa", 5.1, 3.5, 1.4, 0.2),
        ("Typical Iris versicolor", 5.9, 3.0, 4.2, 1.5),
        ("Typical Iris virginica", 6.5, 3.0, 5.5, 1.8),
    ]
    for desc, sl, sw, pl, pw in presets:
        print(f"\n[Test Case] {desc}")
        res = predictor.predict(sl, sw, pl, pw)
        print_prediction_result(res)


def main():
    predictor = IrisPredictor()

    if len(sys.argv) == 5:
        # Command line arguments: python predict.py <sl> <sw> <pl> <pw>
        try:
            sl, sw, pl, pw = [float(x) for x in sys.argv[1:5]]
            res = predictor.predict(sl, sw, pl, pw)
            print_prediction_result(res)
            return
        except ValueError:
            print("Invalid arguments. Usage: python predict.py <sepal_len> <sepal_wid> <petal_len> <petal_wid>")
            return

    # If no arguments, run canonical benchmarks and then ask for user input
    run_benchmark_presets(predictor)

    print("Would you like to test custom measurements? (Enter 'q' to exit)")
    while True:
        try:
            user_in = input("\nEnter 4 values separated by space (SepalL SepalW PetalL PetalW) or 'q': ").strip()
            if user_in.lower() == 'q' or not user_in:
                print("Exiting prediction CLI.")
                break
            vals = [float(x) for x in user_in.split()]
            if len(vals) != 4:
                print("Please provide exactly 4 numeric values.")
                continue
            res = predictor.predict(vals[0], vals[1], vals[2], vals[3])
            print_prediction_result(res)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
