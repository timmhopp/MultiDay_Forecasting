"""
Recursive prediction utilities for long-term forecasting.
"""

import torch
import numpy as np


def predict_long_future_patterns_fused(model, initial_input_window_long: torch.Tensor, 
                                      num_steps_to_predict: int, device: torch.device):
    """
    Perform recursive long-term prediction using FusedODModel.
    
    Args:
        model: Trained FusedODModel
        initial_input_window_long: Initial input window (1, W_long, N, N)
        num_steps_to_predict: Number of steps to predict recursively
        device: Device for computation
        
    Returns:
        predicted_series: Recursive predictions
    """
    model.eval()
    with torch.no_grad():
        current_input_window_long = initial_input_window_long.clone().to(device)
        
        # Check if model is multi-step or single-step
        W_long = model.W_long
        
        if hasattr(model, 'num_prediction_horizons') and model.num_prediction_horizons > 1:
            # Multi-step model
            all_multi_horizon_predictions = []
            
            for _ in range(num_steps_to_predict):
                # Model outputs (1, num_prediction_horizons, N, N)
                multi_horizon_prediction = model(current_input_window_long)
                multi_horizon_prediction = torch.clamp(multi_horizon_prediction, min=1e-6)
                
                # Store the entire multi-horizon prediction
                all_multi_horizon_predictions.append(multi_horizon_prediction.cpu().squeeze(0))
                
                # Use first horizon prediction for recursive update
                next_step_prediction = multi_horizon_prediction[:, 0, :, :].unsqueeze(1)
                current_input_window_long = torch.cat(
                    (current_input_window_long[:, 1:, :, :], next_step_prediction), dim=1
                )
            
            return torch.stack(all_multi_horizon_predictions, dim=0)
        
        else:
            # Single-step model
            predicted_series = []
            
            for _ in range(num_steps_to_predict):
                # Model outputs (1, N, N)
                prediction = model(current_input_window_long)
                prediction = torch.clamp(prediction, min=1e-6)
                
                # Store prediction
                predicted_series.append(prediction.cpu().squeeze(0))
                
                # Update input window
                current_input_window_long = torch.cat(
                    (current_input_window_long[:, 1:, :, :], prediction.unsqueeze(1)), dim=1
                )
            
            return torch.stack(predicted_series, dim=0)