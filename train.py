"""
Training pipeline for Iris Flower Classification Artificial Neural Network (ANN).
Executes end-to-end training, testing, metric computation, model saving, and visualization.
"""

import os
import sys
import json
import time

# Ensure src module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import IrisDataLoader, CLASS_NAMES, SHORT_CLASSES, FEATURE_NAMES
from src.ann_model import ArtificialNeuralNetwork
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer


def main():
    print("=" * 70)
    print("    IRIS FLOWER CLASSIFICATION USING ARTIFICIAL NEURAL NETWORK (ANN)   ")
    print("=" * 70)
    start_time = time.time()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_dir = os.path.join(base_dir, "saved_models")
    results_dir = os.path.join(base_dir, "results")
    model_filepath = os.path.join(model_save_dir, "iris_ann_model.json")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load and prepare dataset
    print("\n[Step 1/5] Loading and Preprocessing Iris Dataset...")
    data_loader = IrisDataLoader()
    data = data_loader.prepare_data(test_size=0.20, random_state=42)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    y_test_labels = data["y_test_labels"]

    print(f"  > Total Samples : 150")
    print(f"  > Training Set  : {X_train.shape[0]} samples (80%)")
    print(f"  > Test Set      : {X_test.shape[0]} samples (20%)")
    print(f"  > Input Features: 4 (Sepal L/W, Petal L/W)")
    print(f"  > Target Classes: 3 ({', '.join(CLASS_NAMES)})")

    # 2. Build Artificial Neural Network
    print("\n[Step 2/5] Initializing Neural Network Architecture...")
    model = ArtificialNeuralNetwork(
        input_dim=4,
        hidden_dims=[16, 8],
        output_dim=3,
        l2_reg=0.001
    )
    print("  > Layer Architecture:")
    print("    - Input Layer     : 4 Neurons")
    print("    - Hidden Layer 1  : 16 Neurons (He Normal Init + ReLU)")
    print("    - Hidden Layer 2  : 8 Neurons (He Normal Init + ReLU)")
    print("    - Output Layer    : 3 Neurons (Xavier Init + Softmax)")
    print("    - Loss Function   : Categorical Cross-Entropy Loss")
    print("    - Optimizer       : Adam (Adaptive Moment Estimation, lr=0.015)")

    # 3. Model Training
    print("\n[Step 3/5] Training ANN Model...")
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=220,
        batch_size=16,
        learning_rate=0.015,
        verbose=True
    )

    # 4. Evaluation on Unseen Test Set
    print("\n[Step 4/5] Evaluating Model on Unseen Test Set...")
    test_preds_proba = model.predict_proba(X_test)
    test_preds_labels = model.predict(X_test)

    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(
        y_true=y_test_labels,
        y_pred=test_preds_labels,
        class_names=CLASS_NAMES
    )

    print("\n" + "=" * 65)
    print("                      CLASSIFICATION REPORT                     ")
    print("=" * 65)
    print(evaluator.classification_report_str(metrics, CLASS_NAMES))

    print("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    cm = metrics["confusion_matrix"]
    print(f"  {'':<18} {'Pred Setosa':<12} {'Pred Versi':<12} {'Pred Virgin':<12}")
    for idx, row in enumerate(cm):
        print(f"  {CLASS_NAMES[idx]:<18} {row[0]:<12} {row[1]:<12} {row[2]:<12}")

    # 5. Saving Model and Visualizations
    print("\n[Step 5/5] Saving Trained Model and Generating Visualizations...")
    metadata = {
        "mean": data["mean"].tolist(),
        "std": data["std"].tolist(),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "short_classes": SHORT_CLASSES,
        "metrics": metrics,
        "history": {
            "epoch": history["epoch"],
            "train_loss": [round(float(v), 5) for v in history["train_loss"]],
            "train_acc": [round(float(v), 5) for v in history["train_acc"]],
            "val_loss": [round(float(v), 5) for v in history["val_loss"]],
            "val_acc": [round(float(v), 5) for v in history["val_acc"]]
        }
    }
    model.save_model(model_filepath, extra_meta=metadata)
    print(f"  [OK] Model & Scaler parameters saved to: {model_filepath}")

    visualizer = Visualizer(output_dir=results_dir)
    p1 = visualizer.plot_training_curves(history)
    p2 = visualizer.plot_confusion_matrix(metrics["confusion_matrix"], SHORT_CLASSES)
    p3 = visualizer.plot_decision_boundary(model, data_loader, data["raw_df"])
    p4 = visualizer.plot_feature_distributions(data["raw_df"])

    print(f"  [OK] Training curves plot saved to       : {p1}")
    print(f"  [OK] Confusion matrix heatmap saved to  : {p2}")
    print(f"  [OK] Decision boundary map saved to     : {p3}")
    print(f"  [OK] Feature distributions saved to     : {p4}")

    total_time = time.time() - start_time
    print(f"\nTraining pipeline completed successfully in {total_time:.2f} seconds!")
    print(f"Final Test Accuracy: {metrics['accuracy'] * 100:.2f}%\n")

if __name__ == "__main__":
    main()
