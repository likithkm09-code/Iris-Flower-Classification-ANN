"""Evaluation metrics for Multi-Class Iris Classification (Confusion Matrix, Precision, Recall, F1)."""

import numpy as np


class ModelEvaluator:
    """Computes comprehensive classification metrics from ground-truth and predictions."""

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """
        Builds a num_classes x num_classes confusion matrix.
        Rows: True classes
        Columns: Predicted classes
        """
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
        """Calculates accuracy, per-class precision, recall, f1-score, and macro/weighted averages."""
        num_classes = len(class_names)
        cm = ModelEvaluator.confusion_matrix(y_true, y_pred, num_classes=num_classes)

        total_samples = len(y_true)
        accuracy = float(np.sum(np.diag(cm)) / total_samples) if total_samples > 0 else 0.0

        per_class = {}
        precisions = []
        recalls = []
        f1_scores = []
        supports = []

        for c in range(num_classes):
            tp = float(cm[c, c])
            fp = float(np.sum(cm[:, c]) - tp)
            fn = float(np.sum(cm[c, :]) - tp)
            tn = float(total_samples - tp - fp - fn)
            support = int(np.sum(cm[c, :]))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            supports.append(support)

            per_class[class_names[c]] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": support
            }

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1_scores))

        weights = np.array(supports) / total_samples if total_samples > 0 else np.ones(num_classes)
        weighted_precision = float(np.sum(np.array(precisions) * weights))
        weighted_recall = float(np.sum(np.array(recalls) * weights))
        weighted_f1 = float(np.sum(np.array(f1_scores) * weights))

        return {
            "accuracy": round(accuracy, 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "macro_avg": {
                "precision": round(macro_precision, 4),
                "recall": round(macro_recall, 4),
                "f1_score": round(macro_f1, 4)
            },
            "weighted_avg": {
                "precision": round(weighted_precision, 4),
                "recall": round(weighted_recall, 4),
                "f1_score": round(weighted_f1, 4)
            },
            "total_samples": total_samples
        }

    @staticmethod
    def classification_report_str(metrics: dict, class_names: list) -> str:
        """Formats metrics dictionary into standard classification report string."""
        header = f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}\n"
        divider = "-" * 62 + "\n"
        rows = []

        for name in class_names:
            cls_m = metrics["per_class"][name]
            rows.append(
                f"{name:<20} {cls_m['precision']:<10.4f} {cls_m['recall']:<10.4f} "
                f"{cls_m['f1_score']:<10.4f} {cls_m['support']:<8d}"
            )

        rows_str = "\n".join(rows) + "\n"
        acc_str = f"\n{'Accuracy':<20} {'':<10} {'':<10} {metrics['accuracy']:<10.4f} {metrics['total_samples']:<8d}\n"
        
        macro = metrics["macro_avg"]
        macro_str = f"{'Macro Avg':<20} {macro['precision']:<10.4f} {macro['recall']:<10.4f} {macro['f1_score']:<10.4f} {metrics['total_samples']:<8d}\n"
        
        weighted = metrics["weighted_avg"]
        weighted_str = f"{'Weighted Avg':<20} {weighted['precision']:<10.4f} {weighted['recall']:<10.4f} {weighted['f1_score']:<10.4f} {metrics['total_samples']:<8d}\n"

        return header + divider + rows_str + divider + acc_str + macro_str + weighted_str
