import sklearn.metrics
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from tqdm import tqdm
import numpy as np

from evaluation.evaluation_structures import PredictionMetadata, OperatingPointMetrics, EvalMetrics


def predict_samples(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> PredictionMetadata:
    """Generates model predictions and collects ground truth labels."""
    model = model.eval()
    pred_probs_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting Samples"):
            inputs, labels = batch  # Assuming dataloader outputs tuples (inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            logits = model(inputs)
            pred_probs = torch.sigmoid(logits)  # Convert logits to probabilities

            # Collect results
            pred_probs_all.append(pred_probs.cpu())  # Move to CPU
            labels_all.append(labels.cpu())         # Move to CPU

    # Concatenate all batches
    pred_probs_all = torch.cat(pred_probs_all, dim=0).squeeze(1)
    labels_all = torch.cat(labels_all, dim=0)

    return PredictionMetadata(
        pred_probs=pred_probs_all.numpy(),
        labels_gt=labels_all.numpy(),
    )

def compute_eval_metrics(prediction_meta: PredictionMetadata) -> EvalMetrics:
    """Computes average precision, precision-recall curve, and metrics at optimal threshold."""
    # Extract predictions and ground truth
    pred_probs = prediction_meta.pred_probs
    labels_gt = prediction_meta.labels_gt

    # Ensure inputs are NumPy arrays and 1-dimensional
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.cpu().numpy()
    if isinstance(labels_gt, torch.Tensor):
        labels_gt = labels_gt.cpu().numpy()

    pred_probs = pred_probs.ravel()  # Flatten to 1D
    labels_gt = labels_gt.ravel()    # Flatten to 1D

    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true=labels_gt, probas_pred=pred_probs)
    thresholds = np.append(thresholds, 1.0)  # Align thresholds with precisions/recalls

    # Compute average precision
    avg_precision = average_precision_score(y_true=labels_gt, y_score=pred_probs)

    # Find the operating point that maximizes F1
    threshold_op_metrics = compute_operating_point_metrics_max_f1(precisions, recalls, thresholds)

    # Compute metrics at the optimal threshold
    metrics_op = compute_operating_point_metrics_at_threshold(precisions, recalls, thresholds, threshold_op_metrics.threshold_op)

    fpr, tpr, roc_thresholds = roc_curve(y_true=labels_gt, y_score=pred_probs)
    roc_auc = roc_auc_score(y_true=labels_gt, y_score=pred_probs)


    return EvalMetrics(
        precisions=torch.tensor(precisions),
        recalls=torch.tensor(recalls),
        thresholds=torch.tensor(thresholds),
        average_precision=avg_precision,
        metrics_op=metrics_op,
        roc_auc=roc_auc,  # Add AUC to your EvalMetrics if desired
        fpr=torch.tensor(fpr),  # Optionally store FPR
        tpr=torch.tensor(tpr),  # Optionally store TPR
    )


def eval_model(
        model: torch.nn.Module,
        dataloader_test: torch.utils.data.DataLoader,
        device: torch.device
) -> EvalMetrics:
    """Given a model and test dataset, evaluates the model on the test dataset."""

    prediction_meta = predict_samples(model, dataloader_test, device)
    return compute_eval_metrics(prediction_meta)


def compute_operating_point_metrics_max_f1(
        precisions: torch.Tensor,
        recalls: torch.Tensor,
        thresholds: torch.Tensor
) -> OperatingPointMetrics:
    """Calculate eval metrics at the operating point (aka threshold) that maximizes F1 score.
    Precisions/recalls/thresholds are computed from: sklearn.metrics.precision_recall_curve()

    Args:
        precisions:
        recalls:
        thresholds:

    Returns:
        operating_point_metrics: eval metrics at the threshold that maximizes the F1 score.

    """
    # BEGIN YOUR CODE
    precision, recall_op, f1_score_op = 0.0, 0.0, 0.0

    for i in range(len(thresholds)):
      metrics = compute_operating_point_metrics_at_threshold(
        precisions,
        recalls,
        thresholds,
        thresholds[i]
      )

      if metrics.f1_score_op > f1_score_op:
        f1_score_op = metrics.f1_score_op
        precision_op = metrics.precision_op
        recall_op = metrics.recall_op
        threshold_op = metrics.threshold_op

    # END YOUR CODE
    return OperatingPointMetrics(
        precision_op=precision_op,
        recall_op=recall_op,
        f1_score_op=f1_score_op,
        threshold_op=threshold_op,
    )


def compute_operating_point_metrics_at_threshold(
        precisions: torch.Tensor,
        recalls: torch.Tensor,
        thresholds: torch.Tensor,
        threshold_op: float,
) -> OperatingPointMetrics:
    """Compute eval metrics at a specific input threshold.
    Precisions/recalls/thresholds are computed from: sklearn.metrics.precision_recall_curve()

    Args:
        precisions:
        recalls:
        thresholds:
        threshold_op: Threshold to calculate precision/recall/f1 for.
            Note that `threshold_op` will in general not be exactly in `thresholds`.
            In this case, use the precision/recall values corresponding to the first threshold
            in `thresholds` where `threshold >= threshold_op`.

    Returns:
        operating_point_metrics: Eval metrics at the given threshold (`threshold_op`).
    """
    # BEGIN YOUR CODE
    precision, recall, f1 = 0.0, 0.0, 0.0

    for i, threshold in enumerate(thresholds):
      if threshold >= threshold_op:
        precision = precisions[i]
        recall = recalls[i]
        if ((precision + recall) <= 0):
          f1 = 0
        else: f1 = 2 * (precision * recall) / (precision + recall)
        break
    # END YOUR CODE
    return OperatingPointMetrics(
              precision_op=precision,
              recall_op=recall,
              f1_score_op=f1,
              threshold_op=threshold,
            )