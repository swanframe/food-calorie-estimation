from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_true_vs_pred(
    y_true,
    y_pred,
    title: str = "",
    model_name: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Scatter plot: true vs predicted values with y=x reference line.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1.5)

    label = model_name if model_name is not None else ""
    plt.xlabel("True calories (kCal)")
    plt.ylabel("Predicted calories (kCal)")
    plt.title(title or f"True vs Predicted Calories {label}".strip())

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_error_histogram(
    y_true,
    y_pred,
    title: str = "",
    model_name: Optional[str] = None,
    save_path: Optional[Path] = None,
    bins: int = 30,
) -> None:
    """
    Histogram of prediction errors (pred - true).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true

    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=bins)
    label = model_name if model_name is not None else ""
    plt.xlabel("Prediction error (kCal) [pred - true]")
    plt.ylabel("Count")
    plt.title(title or f"Error distribution {label}".strip())
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()