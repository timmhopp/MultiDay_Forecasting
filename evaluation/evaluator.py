"""
Evaluation utilities for FusedODModel.
"""

import torch
import numpy as np

from ..training.data_loaders import build_windows_generator_single_step_fused, build_windows_generator_multi_step_fused
from ..utils.metrics import compute_metrics


def eval_fused_model(model, series_OD: torch.Tensor, W_long: int, W_short: int, bs: int, device: torch.device):
    """
    Evaluate FusedODModel for single-step prediction.
    
    Args:
        model: Trained FusedODModel
        series_OD: Test data tensor (T, N, N)
        W_long: Long-term window length
        W_short: Short-term window length
        bs: Batch size
        device: Device for evaluation
        
    Returns:
        Yt: True values
        Yp_counts: Predicted values
        metrics: Evaluation metrics
    """
    model.eval()
    all_preds, all_trues = [], []

    test_data_generator = build_windows_generator_single_step_fused(series_OD, W_long, W_short, bs)

    try:
        from tqdm.auto import tqdm
        T, N_dummy, _ = series_OD.shape
        num_windows_test = max(0, T - W_long)
        num_batches_test = (num_windows_test + bs - 1) // bs
        batch_iterator = tqdm(test_data_generator, total=num_batches_test, desc="Evaluating Fused Model")
    except ImportError:
        batch_iterator = test_data_generator

    with torch.no_grad():
        for xb_long, xb_short, yb in batch_iterator:
            xb_long, yb = xb_long.to(device), yb.to(device)
            p = model(xb_long)  # FusedODModel takes only xb_long
            all_preds.append(p.cpu().numpy())
            all_trues.append(yb.cpu().numpy())

    Yp_counts = np.concatenate(all_preds, axis=0)
    Yt = np.concatenate(all_trues, axis=0)

    Yp_counts = np.maximum(Yp_counts, 1e-6)

    metrics = compute_metrics(Yt, Yp_counts)

    return Yt, Yp_counts, metrics


def eval_fused_model_multi_step(model, series_OD: torch.Tensor, W_long: int, prediction_horizons: list, bs: int, device: torch.device):
    """
    Evaluate FusedODModel for multi-step prediction.
    
    Args:
        model: Trained FusedODModel
        series_OD: Test data tensor (T, N, N)
        W_long: Long-term window length
        prediction_horizons: List of prediction horizons
        bs: Batch size
        device: Device for evaluation
        
    Returns:
        final_trues_by_horizon: True values by horizon
        final_preds_by_horizon: Predicted values by horizon  
        metrics_by_horizon: Evaluation metrics by horizon
    """
    model.eval()

    num_prediction_horizons = len(prediction_horizons)

    # Initialize lists for each horizon
    all_preds_by_horizon = [[] for _ in range(num_prediction_horizons)]
    all_trues_by_horizon = [[] for _ in range(num_prediction_horizons)]

    test_data_generator = build_windows_generator_multi_step_fused(series_OD, W_long, prediction_horizons, bs)

    # Calculate total number of windows for progress bar
    T, N_dummy, _ = series_OD.shape
    max_horizon = max(prediction_horizons)
    num_windows_test = max(0, T - W_long - max_horizon + 1)
    num_batches_test = (num_windows_test + bs - 1) // bs

    try:
        from tqdm.auto import tqdm
        batch_iterator = tqdm(test_data_generator, total=num_batches_test, desc="Evaluating Fused Multi-Step Model")
    except ImportError:
        batch_iterator = test_data_generator

    with torch.no_grad():
        for xb_long, yb_multi_step in batch_iterator:
            xb_long = xb_long.to(device)

            pred_multi_step = model(xb_long)  # (B, num_horizons, N, N)
            pred_multi_step = torch.clamp(pred_multi_step, min=1e-6)

            # Move to CPU and convert to numpy
            pred_np = pred_multi_step.cpu().numpy()
            true_np = yb_multi_step.cpu().numpy()

            for h_idx in range(num_prediction_horizons):
                all_preds_by_horizon[h_idx].append(pred_np[:, h_idx, :, :])
                all_trues_by_horizon[h_idx].append(true_np[:, h_idx, :, :])

    # Concatenate results for each horizon
    final_preds_by_horizon = [np.concatenate(preds_list, axis=0) for preds_list in all_preds_by_horizon]
    final_trues_by_horizon = [np.concatenate(trues_list, axis=0) for trues_list in all_trues_by_horizon]

    # Compute metrics for each horizon
    metrics_by_horizon = {}
    for h_idx, horizon in enumerate(prediction_horizons):
        y_true_horizon = final_trues_by_horizon[h_idx]
        y_pred_horizon = final_preds_by_horizon[h_idx]
        metrics_by_horizon[f"Horizon_{horizon}steps"] = compute_metrics(y_true_horizon, y_pred_horizon)

    return final_trues_by_horizon, final_preds_by_horizon, metrics_by_horizon