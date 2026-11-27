import numpy as np


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "bias": float(np.mean(error)),
    }
