"""Scoring functions shared by every model, so comparisons are fair."""

from sklearn.metrics import brier_score_loss, log_loss


def evaluate(y_true, p_pred) -> dict:
    """Score predicted win probabilities against actual outcomes.

    y_true: actual labels (1 = offense won, 0 = lost)
    p_pred: predicted probability that the offense wins, between 0 and 1
    Lower is better for both metrics.
    """
    return {
        "log_loss": log_loss(y_true, p_pred),
        "brier": brier_score_loss(y_true, p_pred),
    }