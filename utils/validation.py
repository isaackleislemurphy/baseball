""" """

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def score_regression(
    yhat_train: np.ndarray,
    y_train: np.ndarray,
    yhat_test: np.ndarray,
    y_test: np.ndarray,
    w_train: Optional[np.ndarray] = None,
    w_test: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Bare-bones regression scoring: MAE, RMSE, Pearson, Spearman, and bias.

    Parameters
    ----------
    yhat_test : np.ndarray
        OOS predictions
    y_test : np.ndarray
        OOS observed values
    yhat_train : np.ndarray
        IS predictions
    y_train : np.ndarray
        IS observed values
    w_test : np.ndarray, optional
        OOS sample weights, if applicable
    w_train : np.ndarray, optional
        IS sample weights, if applicable

    Returns : pd.DataFrame
        A dataframe of basic regression scoring values. Indices are metrics,
        columns are train/test.
    """
    scoring = {}

    # 1.) MAE
    scoring["mae"] = (
        mean_absolute_error(y_test, yhat_test, sample_weight=w_test),
        mean_absolute_error(y_train, yhat_train, sample_weight=w_train),
    )

    # 2.) RMSE
    scoring["rmse"] = (
        mean_squared_error(y_test, yhat_test, sample_weight=w_test) ** 0.5,
        mean_squared_error(y_train, yhat_train, sample_weight=w_train) ** 0.5,
    )

    # 3.) Pearson
    scoring["r2"] = (
        r2_score(y_test, yhat_test, sample_weight=w_test) ** 0.5,
        r2_score(y_train, yhat_train, sample_weight=w_train) ** 0.5,
    )

    # 4.) Spearman
    scoring["spearman"] = (
        spearmanr(y_test, yhat_test)[0],
        spearmanr(y_train, yhat_train)[0],
    )

    # 5.) Bias
    scoring["bias"] = (
        np.average(yhat_test - y_test, weights=w_test),
        np.average(yhat_train - y_train, weights=w_train),
    )

    return pd.DataFrame(scoring, index=["out_of_sample", "in_sample"]).T


def score_classification(
    yhat_proba_train: np.ndarray,
    y_train: np.ndarray,
    yhat_proba_test: np.ndarray,
    y_test: np.ndarray,
    w_train: Optional[np.ndarray] = None,
    w_test: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Bare-bones probabilistic classification scoring.

    Metrics:
    - Log loss (cross-entropy)
    - Brier score (multiclass)
    - Accuracy (argmax)
    - Macro ROC-AUC (OVR)
    - Expected Calibration Error (ECE)

    Parameters
    ----------
    yhat_proba_* : np.ndarray
        Predicted class probabilities, shape (n_samples, n_classes)
    y_* : np.ndarray
        True class labels, integer encoded
    w_* : np.ndarray, optional
        Sample weights

    Returns
    -------
    pd.DataFrame
        Rows = metrics, columns = out_of_sample / in_sample
    """
    scoring = {}

    # 1.) Log loss (proper scoring rule)
    scoring["log_loss"] = (
        log_loss(y_test, yhat_proba_test, sample_weight=w_test),
        log_loss(y_train, yhat_proba_train, sample_weight=w_train),
    )

    # 2.) Brier score (multiclass)
    def multiclass_brier(y, p, w=None):
        y_onehot = np.eye(p.shape[1])[y]
        sq_err = np.sum((p - y_onehot) ** 2, axis=1)
        return np.average(sq_err, weights=w)

    scoring["brier"] = (
        multiclass_brier(y_test, yhat_proba_test, w_test),
        multiclass_brier(y_train, yhat_proba_train, w_train),
    )

    # 3.) Accuracy (argmax; not a proper scoring rule, but useful)
    scoring["accuracy"] = (
        accuracy_score(y_test, np.argmax(yhat_proba_test, axis=1), sample_weight=w_test),
        accuracy_score(y_train, np.argmax(yhat_proba_train, axis=1), sample_weight=w_train),
    )

    # 4.) Macro ROC-AUC (OVR)
    scoring["roc_auc_macro"] = (
        roc_auc_score(
            y_test,
            yhat_proba_test,
            average="macro",
            multi_class="ovr",
            sample_weight=w_test,
        ),
        roc_auc_score(
            y_train,
            yhat_proba_train,
            average="macro",
            multi_class="ovr",
            sample_weight=w_train,
        ),
    )

    # 5.) Expected Calibration Error (ECE)
    def ece(y, p, n_bins=10, w=None):
        conf = np.max(p, axis=1)
        pred = np.argmax(p, axis=1)
        correct = (pred == y).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if not np.any(mask):
                continue
            bin_weight = np.sum(w[mask]) if w is not None else np.sum(mask)
            acc = np.average(correct[mask], weights=w[mask] if w is not None else None)
            avg_conf = np.average(conf[mask], weights=w[mask] if w is not None else None)
            ece += bin_weight * np.abs(acc - avg_conf)

        norm = np.sum(w) if w is not None else len(y)
        return ece / norm

    scoring["ece"] = (
        ece(y_test, yhat_proba_test, w=w_test),
        ece(y_train, yhat_proba_train, w=w_train),
    )

    return pd.DataFrame(scoring, index=["out_of_sample", "in_sample"]).T
