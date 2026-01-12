"""
Data loader utilities for building training/evaluation windows.
"""

import torch
import numpy as np


def build_windows_generator_single_step_fused(series_OD: torch.Tensor, W_long: int, W_short: int, batch_size: int):
    """
    Generator for single-step prediction windows for FusedODModel.
    
    Args:
        series_OD: Time series tensor (T, N, N)
        W_long: Long-term window length
        W_short: Short-term window length  
        batch_size: Batch size
        
    Yields:
        X_batch_long: Long-term input batch (B, W_long, N, N)
        X_batch_short: Short-term input batch (B, W_short, N, N)
        Y_batch: Target batch (B, N, N)
    """
    if series_OD.is_cuda:
        series_OD = series_OD.cpu()

    T, N, _ = series_OD.shape

    # The effective window size for generating data points is W_long
    num_windows = max(0, T - W_long)
    num_batches = (num_windows + batch_size - 1) // batch_size  # Ceiling division

    for i in range(num_batches):
        start_window_idx = i * batch_size
        end_window_idx = min(num_windows, start_window_idx + batch_size)

        X_batch_long = []
        X_batch_short = []
        Y_batch = []

        for t in range(start_window_idx, end_window_idx):
            # Long-term input window: [t, t + W_long - 1]
            X_batch_long.append(series_OD[t : t + W_long])

            # Short-term input window: [t + W_long - W_short, t + W_long - 1]
            short_start_idx = max(0, t + W_long - W_short)
            X_batch_short.append(series_OD[short_start_idx : t + W_long])

            # Target is the single next OD matrix: [t + W_long]
            Y_batch.append(series_OD[t + W_long])

        X_batch_long = torch.stack(X_batch_long, axis=0).float()
        X_batch_short = torch.stack(X_batch_short, axis=0).float()
        Y_batch = torch.stack(Y_batch, axis=0).float()

        yield X_batch_long, X_batch_short, Y_batch


def build_windows_generator_multi_step_fused(series_OD: torch.Tensor, W_long: int, prediction_horizons: list, batch_size: int):
    """
    Generator for multi-step prediction windows for FusedODModel.
    
    Args:
        series_OD: Time series tensor (T, N, N)
        W_long: Long-term window length
        prediction_horizons: List of prediction horizons
        batch_size: Batch size
        
    Yields:
        X_batch_long: Long-term input batch (B, W_long, N, N)
        Y_batch_multi_step: Multi-step target batch (B, num_horizons, N, N)
    """
    if series_OD.is_cuda:
        series_OD = series_OD.cpu()

    T, N, _ = series_OD.shape

    # Determine the maximum horizon needed
    max_horizon = max(prediction_horizons)

    # Account for both input window and furthest prediction horizon
    num_windows = max(0, T - W_long - max_horizon + 1)
    num_batches = (num_windows + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_window_idx = i * batch_size
        end_window_idx = min(num_windows, start_window_idx + batch_size)

        X_batch_long = []
        Y_batch_multi_step = []

        for t in range(start_window_idx, end_window_idx):
            # Long-term input window: [t, t + W_long - 1]
            X_batch_long.append(series_OD[t : t + W_long])

            # Collect targets for all prediction horizons
            current_horizons_targets = []
            for horizon in prediction_horizons:
                # Target at t + W_long + horizon - 1
                target_idx = t + W_long + horizon - 1
                current_horizons_targets.append(series_OD[target_idx])

            Y_batch_multi_step.append(torch.stack(current_horizons_targets, dim=0))

        X_batch_long = torch.stack(X_batch_long, axis=0).float()
        Y_batch_multi_step = torch.stack(Y_batch_multi_step, axis=0).float()

        yield X_batch_long, Y_batch_multi_step