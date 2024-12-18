import dataclasses
from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class PredictionMetadata:
    """Stores the output of running inference on an evaluation dataset.
    """
    # Predicted probabilities.
    # shape=[num_rows]. Probability that i-th example is a landslide (positive class).
    pred_probs: torch.Tensor

    # Ground-truth labels.
    # shape=[num_rows]. 0 means "nno landslide", 1 means "landslide".
    labels_gt: torch.Tensor


@dataclass
class OperatingPointMetrics:
    """These are metrics for a given operating point (aka classification threshold), where the classification rule is:
        "is_faulty" if pred_prob >= threshold_op else "not_faulty"
    """
    # precision/recall/f1 at the given classification threshold
    precision_op: float
    recall_op: float
    f1_score_op: float
    # Classification threshold
    threshold_op: float

    def summarize_to_str(self) -> str:
        return f"(T={self.threshold_op:.5f}) precision@T={self.precision_op:.5f} recall@T={self.recall_op:.5f} f1@T={self.f1_score_op:.5f}"

    def to_pydict(self) -> Dict[str, Any]:
        """Convert to a python-friendly dict format.
        Similar in spirit to dataclasses.asdict(self), but also convert non-built-in types like
        torch.Tensor to similar built-in types.

        Returns:

        """
        return dataclasses.asdict(self)


@dataclass
class EvalMetrics:
    """Evaluation metrics and metadata.
    """
    # precisions/recalls/thresholds for the eval set.
    # Tip: see: sklearn.metrics.precision_recall_curve
    # shape=[num_rows + 1]
    precisions: torch.Tensor
    # recalls, decreasing. shape=[num_rows + 1]
    recalls: torch.Tensor
    # thresholds, increasing. shape=[num_rows]
    thresholds: torch.Tensor

    # threshold-agnostic score. similar (but not exactly) to area-under-the-curve of the precision-recall curve.
    # Tip: see: sklearn.metrics.average_precision_score
    average_precision: float

    # ROC-AUC metrics
    roc_auc: float  # ROC-AUC score
    fpr: torch.Tensor  # False Positive Rate
    tpr: torch.Tensor  # True Positive Rate

    # Metrics at a specific operating point (threshold that maximizes f1)
    metrics_op: OperatingPointMetrics

    def summarize_to_str(self) -> str:
        return (f"AP={self.average_precision:.5f} ROC-AUC={self.roc_auc:.5f} "
                f"{self.metrics_op.summarize_to_str()}")

    def to_pydict(self) -> Dict[str, Any]:
        """Convert to a python-friendly dict format.
        Similar in spirit to dataclasses.asdict(self), but also convert non-built-in types like
        torch.Tensor to similar built-in types.

        Returns:

        """
        pydict = dataclasses.asdict(self)
        pydict["precisions"] = pydict["precisions"].tolist()
        pydict["recalls"] = pydict["recalls"].tolist()
        pydict["thresholds"] = pydict["thresholds"].tolist()
        pydict["fpr"] = pydict["fpr"].tolist()
        pydict["tpr"] = pydict["tpr"].tolist()
        return pydict