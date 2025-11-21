from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(
    y_true,
    y_pred,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute common regression metrics for calorie prediction.

    Args:
        y_true: array-like of true values.
        y_pred: array-like of predicted values.
        eps:   small value to avoid division by zero in MAPE.

    Returns:
        dict with keys: 'mae', 'mse', 'rmse', 'mape', 'r2'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAPE: mean absolute percentage error (%)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0

    # R²: coefficient of determination
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def print_regression_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Nicely print metrics dict produced by compute_regression_metrics.
    """
    p = prefix
    print(f"{p}MAE:  {metrics['mae']:.2f} kCal")
    print(f"{p}RMSE: {metrics['rmse']:.2f} kCal")
    print(f"{p}MSE:  {metrics['mse']:.2f}")
    print(f"{p}MAPE: {metrics['mape']:.2f} %")
    print(f"{p}R²:   {metrics['r2']:.3f}")