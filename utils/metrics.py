"""
Evaluation metrics for OD prediction models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, eps=1e-6):
    """
    Compute evaluation metrics for OD prediction.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        eps: Small epsilon to avoid division by zero
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    yt, yp = y_true.ravel(), y_pred.ravel()
    
    # Basic metrics
    mse = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    
    # MAPE (Mean Absolute Percentage Error)
    mask = np.abs(yt) > eps
    mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100 if mask.any() else np.nan
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + eps)) * 100
    
    # R-squared
    r2 = r2_score(yt, yp)
    
    return {
        "MSE": mse,
        "RMSE": rmse, 
        "MAE": mae,
        "MAPE(%)": mape,
        "SMAPE(%)": smape,
        "R2": r2
    }