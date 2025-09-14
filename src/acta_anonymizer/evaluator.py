"""NER evaluation metrics."""

from typing import Dict, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, classification_report, f1_score


class Evaluator:
    """Evaluate NER model performance."""

    def __init__(self, id_to_label: Dict[int, str]):
        self.id_to_label = id_to_label

    def evaluate(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

        # Convert predictions to probabilities and get predicted labels
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            true_preds = []
            true_labs = []

            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:  # Not a special token
                    true_preds.append(self.id_to_label.get(pred_id, "O"))
                    true_labs.append(self.id_to_label.get(label_id, "O"))

            true_predictions.append(true_preds)
            true_labels.append(true_labs)

        # Calculate metrics
        results = {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

        # Get detailed classification report
        report = classification_report(true_labels, true_predictions, output_dict=True)

        if "weighted avg" in report:
            results["precision"] = report["weighted avg"]["precision"]
            results["recall"] = report["weighted avg"]["recall"]

        return results
