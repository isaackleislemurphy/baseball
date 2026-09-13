""" """

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


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


def plot_calibration_curve_multiclass(y_true, y_prob, n_bins=10, strategy="uniform", classes=None):
    """
    Plots calibration curves for binary or multiclass problems.

    Parameters:
    - y_true: True labels (1D array).
    - y_prob: Predicted probabilities.
              - If binary, can be 1D (prob of positive class) or 2D (N, 2).
              - If multiclass, must be 2D (N, n_classes).
    - n_bins: Number of bins.
    - strategy: 'uniform' or 'quantile'.
    - classes: List of class names (optional). If None, uses indices 0, 1, ...
    """

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    # --- Logic to handle shapes ---
    plot_list = []

    # Case 1: 1D Array (Binary, prob of positive class)
    if y_prob.ndim == 1:
        plot_list = [(y_true, y_prob, "Positive Class")]

    # Case 2: 2D Array
    elif y_prob.ndim == 2:
        n_classes = y_prob.shape[1]

        # Binary (N, 2) - Standard sklearn behavior is col 0=neg, col 1=pos
        if n_classes == 2:
            plot_list = [(y_true, y_prob[:, 1], "Class 1")]

        # Multiclass (N, >2)
        else:
            if classes is None:
                classes = [f"Class {i}" for i in range(n_classes)]

            # Binarize y_true for One-vs-Rest calculation
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            for i in range(n_classes):
                # y_true_bin[:, i] is 1 if sample is class i, else 0
                plot_list.append((y_true_bin[:, i], y_prob[:, i], classes[i]))

    # --- Plotting ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Reference line
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated", linewidth=1)

    # Loop through each class configuration prepared above
    for y_t, y_p, label in plot_list:

        # 1. Calculate Calibration Curve
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy=strategy)

        # 2. Calculate Brier Score
        bs = brier_score_loss(y_t, y_p)

        # 3. Plot Curve
        line_label = f"{label} (Brier: {bs:.3f})"
        ax[0].plot(prob_pred, prob_true, marker="o", linewidth=2, label=line_label)

        # 4. Plot Histogram
        # Use 'step' style so multiple histograms don't block each other
        ax[1].hist(y_p, range=(0, 1), bins=n_bins, density=False, histtype="step", linewidth=2, label=label, alpha=0.8)

    # Formatting Top Plot
    ax[0].set_ylabel("Fraction of Positives")
    ax[0].set_ylim([-0.05, 1.05])
    ax[0].legend(loc="upper left")
    ax[0].set_title("Calibration Curves")
    ax[0].grid(True, linestyle=":", alpha=0.6)

    # Formatting Bottom Plot
    ax[1].set_xlabel("Mean Predicted Probability")
    ax[1].set_ylabel("Count (Log Scale)")
    # ax[1].set_yscale("log")  # Log scale helps see small classes
    ax[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
