"""Visualization module for ANN Training Curves, Confusion Matrix, and Decision Boundaries."""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Visualizer:
    """Generates high-resolution publication-quality plots."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "results")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Style configurations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        self.palette = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0"]

    def plot_training_curves(self, history: dict, save_filename: str = "training_curves.png") -> str:
        """Plots Training & Validation Loss and Accuracy side by side."""
        epochs = history["epoch"]
        train_loss = history["train_loss"]
        train_acc = history["train_acc"]
        has_val = len(history.get("val_loss", [])) > 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Loss Plot
        ax1.plot(epochs, train_loss, label="Training Loss", color="#4361EE", linewidth=2.2)
        if has_val:
            ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#F72585", linewidth=2.0, linestyle="--")
        ax1.set_title("Categorical Cross-Entropy Loss vs Epochs", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend(frameon=True, facecolor="white", loc="upper right")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Accuracy Plot
        ax2.plot(epochs, [a * 100 for a in train_acc], label="Training Accuracy", color="#2EC4B6", linewidth=2.2)
        if has_val:
            ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Accuracy", color="#E71D36", linewidth=2.0, linestyle="--")
        ax2.set_title("Classification Accuracy vs Epochs (%)", fontsize=13, fontweight="bold", pad=12)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_ylim([0, 105])
        ax2.legend(frameon=True, facecolor="white", loc="lower right")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("Artificial Neural Network (ANN) Training Dynamics", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_confusion_matrix(self, cm, class_names: list, save_filename: str = "confusion_matrix.png") -> str:
        """Plots annotated Confusion Matrix heatmap."""
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        # Ensure cm is a numpy array
        cm = np.array(cm, dtype=np.int64)

        # Compute percentage string annotations
        total_per_row = np.sum(cm, axis=1)[:, np.newaxis]
        total_per_row = np.where(total_per_row == 0, 1, total_per_row)
        cm_norm = cm.astype(float) / total_per_row
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                annotations[i, j] = f"{count}\n({pct:.1f}%)"

        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            linewidths=1.5,
            linecolor="white",
            ax=ax
        )

        ax.set_title("ANN Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Species", fontsize=12, fontweight="semibold", labelpad=10)
        ax.set_ylabel("Actual Species", fontsize=12, fontweight="semibold", labelpad=10)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_decision_boundary(self, model, scaler, raw_df, save_filename: str = "decision_boundary.png") -> str:
        """
        Visualizes ANN Decision Boundary in 2D using Petal Length vs Petal Width
        (the two most discriminative features for Iris species).
        """
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        x_col = "petal_length"
        y_col = "petal_width"

        x_min, x_max = raw_df[x_col].min() - 0.5, raw_df[x_col].max() + 0.5
        y_min, y_max = raw_df[y_col].min() - 0.5, raw_df[y_col].max() + 0.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        # Fix sepal length and sepal width to their means
        sepal_l_mean = scaler.mean_[0]
        sepal_w_mean = scaler.mean_[1]

        # Prepare grid points
        grid_points = np.c_[
            np.full(xx.ravel().shape, sepal_l_mean),
            np.full(xx.ravel().shape, sepal_w_mean),
            xx.ravel(),
            yy.ravel()
        ]

        # Standardize using learned mean and std
        grid_points_scaled = (grid_points - scaler.mean_) / scaler.std_
        probs = model.predict_proba(grid_points_scaled)
        Z = np.argmax(probs, axis=1).reshape(xx.shape)

        # Plot contour decision zones
        cmap_bg = matplotlib.colors.ListedColormap(["#C7D2FE", "#BBF7D0", "#FED7AA"])
        ax.contourf(xx, yy, Z, alpha=0.55, cmap=cmap_bg)
        ax.contour(xx, yy, Z, colors="grey", linewidths=0.7, alpha=0.7)

        # Plot actual scatter points
        species_colors = {"setosa": "#3730A3", "versicolor": "#15803D", "virginica": "#C2410C"}
        for sp, color in species_colors.items():
            subset = raw_df[raw_df["species"].str.lower() == sp]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                c=color,
                label=f"Iris {sp}",
                edgecolors="white",
                linewidth=0.8,
                s=55,
                alpha=0.9
            )

        ax.set_title("ANN Non-Linear Decision Boundaries (Petal Length vs Width)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Petal Length (cm)", fontsize=11)
        ax.set_ylabel("Petal Width (cm)", fontsize=11)
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_feature_distributions(self, df, save_filename: str = "feature_distributions.png") -> str:
        """Plots distribution of all 4 features grouped by species."""
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        titles = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
        axes = axes.flatten()

        custom_palette = {"setosa": "#4361EE", "versicolor": "#2EC4B6", "virginica": "#E71D36"}

        for i, (feat, title) in enumerate(zip(features, titles)):
            sns.boxplot(
                data=df,
                x="species",
                y=feat,
                hue="species",
                legend=False,
                palette=custom_palette,
                ax=axes[i],
                boxprops=dict(alpha=0.8)
            )
            axes[i].set_title(title, fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Species", fontsize=10)
            axes[i].set_ylabel(title, fontsize=10)
            axes[i].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("Iris Morphological Feature Distributions by Species", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
