#!/usr/bin/env python3
"""
Main evaluation script for FusedODModel.

This script handles evaluation of the trained spatiotemporal OD prediction model,
including single-step evaluation, multi-step evaluation, and recursive forecasting.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import model components
from models.fused_od_model import FusedODModel
from models.gat_layer import GraphAttentionLayer
from models.temporal_attention import TemporalAttention

# Import evaluation utilities
from evaluation.evaluator import eval_fused_model, eval_fused_model_multi_step
from evaluation.recursive_predictor import predict_long_future_patterns_fused

# Import visualization and utilities
from utils.visualization import plot_od_heatmaps, plot_scatter, plot_loss_curve
from utils.metrics import compute_metrics
from utils.data_utils import load_adjacency_distance_matrices, load_trip_tensor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate FusedODModel for OD prediction")
    
    # Data paths
    parser.add_argument("--adjacency-path", type=str, required=True,
                       help="Path to adjacency matrix CSV file")
    parser.add_argument("--distance-path", type=str, required=True,
                       help="Path to distance matrix CSV file")
    parser.add_argument("--trips-tensor-path", type=str, required=True,
                       help="Path to trips tensor PT file")
    
    # Model checkpoint
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to model checkpoint")
    
    # Model parameters (must match training)
    parser.add_argument("--hidden-size", type=int, default=64,
                       help="Hidden size for model layers")
    parser.add_argument("--w-long", type=int, default=144,
                       help="Long-term input window length")
    parser.add_argument("--w-short", type=int, default=36,
                       help="Short-term input window length")
    parser.add_argument("--chunk-size-short", type=int, default=9,
                       help="Size of each short-term chunk")
    parser.add_argument("--num-chunks-short", type=int, default=4,
                       help="Number of short-term chunks")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Train/test split ratio")
    
    # Multi-step evaluation
    parser.add_argument("--multi-step", action="store_true",
                       help="Enable multi-step evaluation")
    parser.add_argument("--prediction-horizons", nargs="+", type=int, 
                       default=[1, 36, 144, 432, 1008],
                       help="Prediction horizons for multi-step evaluation")
    
    # Recursive forecasting
    parser.add_argument("--recursive-forecast", action="store_true",
                       help="Enable recursive long-term forecasting")
    parser.add_argument("--forecast-steps", type=int, default=144,
                       help="Number of steps for recursive forecasting")
    parser.add_argument("--plot-horizons", nargs="+", type=int, 
                       default=[1, 6, 18, 72, 144],
                       help="Horizons to plot for recursive forecasting")
    
    # Output paths
    parser.add_argument("--output-dir", type=str, default="./evaluation_outputs",
                       help="Directory to save evaluation outputs")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    # Evaluation modes
    parser.add_argument("--eval-train", action="store_true",
                       help="Evaluate on training set")
    parser.add_argument("--eval-test", action="store_true", default=True,
                       help="Evaluate on test set")
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup compute device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path, model_config, device):
    """Load model from checkpoint."""
    model = FusedODModel(**model_config).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def evaluate_dataset(model, dataset_tensor, dataset_name, args, device, output_dir):
    """Evaluate model on a dataset."""
    print(f"\nEvaluating on {dataset_name} set...")
    
    if args.multi_step:
        # Multi-step evaluation
        y_true, y_pred, metrics = eval_fused_model_multi_step(
            model, dataset_tensor, args.w_long, args.prediction_horizons, 
            args.batch_size, device
        )
        
        print(f"{dataset_name} set metrics (multi-step):")
        for horizon_key, horizon_metrics in metrics.items():
            print(f"  {horizon_key}:")
            for k, v in horizon_metrics.items():
                print(f"    {k}: {v:.4f}")
        
        # Save results
        results_path = output_dir / f"{dataset_name.lower()}_evaluation_multi_step.npz"
        save_data = {
            'y_true_by_horizon': y_true,
            'y_pred_by_horizon': y_pred,
            'prediction_horizons': args.prediction_horizons,
            'metrics': metrics
        }
        
    else:
        # Single-step evaluation
        y_true, y_pred, metrics = eval_fused_model(
            model, dataset_tensor, args.w_long, args.w_short, 
            args.batch_size, device
        )
        
        print(f"{dataset_name} set metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save results
        results_path = output_dir / f"{dataset_name.lower()}_evaluation_single_step.npz"
        save_data = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    np.savez(results_path, **save_data)
    print(f"{dataset_name} evaluation results saved to {results_path}")
    
    return y_true, y_pred, metrics


def perform_recursive_forecasting(model, test_tensor, args, device, output_dir):
    """Perform recursive long-term forecasting."""
    print(f"\nPerforming recursive forecasting for {args.forecast_steps} steps...")
    
    model.eval()
    
    # Find valid starting point
    earliest_valid_start_idx = 0
    required_total_length = args.w_long + args.forecast_steps
    
    if earliest_valid_start_idx + required_total_length > len(test_tensor):
        print(f"Not enough test data for recursive forecasting. "
              f"Requires {required_total_length} points, but test set has {len(test_tensor)} points.")
        return
    
    with torch.no_grad():
        # Prepare initial input window
        initial_input_window = test_tensor[
            earliest_valid_start_idx : earliest_valid_start_idx + args.w_long
        ].unsqueeze(0)
        
        # Generate predictions
        if args.multi_step:
            # For multi-step models
            predicted_series = predict_long_future_patterns_fused(
                model, initial_input_window, args.forecast_steps, device
            )
        else:
            # For single-step models - need to implement recursive prediction
            predicted_series = predict_long_future_patterns_fused(
                model, initial_input_window, args.forecast_steps, device
            )
        
        # Convert to numpy
        predicted_series_np = predicted_series.cpu().numpy()
        predicted_series_np = np.maximum(predicted_series_np, 0)  # Ensure non-negative
        
        # Get true values for comparison
        true_series = test_tensor[
            earliest_valid_start_idx + args.w_long : 
            earliest_valid_start_idx + args.w_long + args.forecast_steps
        ].cpu().numpy()
        
        print(f"Generated {args.forecast_steps} recursive predictions")
        
        # Plot results for selected horizons
        for h_step in args.plot_horizons:
            if h_step <= args.forecast_steps:
                h_idx = h_step - 1  # Convert to 0-based index
                
                if args.multi_step and predicted_series_np.ndim == 4:
                    # Multi-step model: (steps, horizons, N, N)
                    # Use first horizon from first prediction step
                    pred_od = predicted_series_np[0, 0, :, :]
                else:
                    # Single-step model: (steps, N, N)
                    pred_od = predicted_series_np[h_idx, :, :]
                
                true_od = true_series[h_idx, :, :]
                
                print(f"\nPlotting for horizon {h_step} steps:")
                
                # Save plots
                heatmap_path = output_dir / f"heatmap_horizon_{h_step}.png"
                scatter_path = output_dir / f"scatter_horizon_{h_step}.png"
                
                plot_od_heatmaps(true_od, pred_od, save_path=str(heatmap_path))
                plot_scatter(true_od, pred_od, save_path=str(scatter_path))
                
                print(f"  Heatmap saved to {heatmap_path}")
                print(f"  Scatter plot saved to {scatter_path}")
        
        # Save recursive forecasting results
        forecast_results_path = output_dir / "recursive_forecast_results.npz"
        np.savez(
            forecast_results_path,
            predicted_series=predicted_series_np,
            true_series=true_series,
            forecast_steps=args.forecast_steps,
            plot_horizons=args.plot_horizons,
            w_long=args.w_long,
            start_idx=earliest_valid_start_idx
        )
        print(f"Recursive forecast results saved to {forecast_results_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    # Check file existence
    required_files = [args.adjacency_path, args.distance_path, args.trips_tensor_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
    
    print("Loading data...")
    
    # Load adjacency and distance matrices
    adj_matrix_tensor, dist_matrix_tensor = load_adjacency_distance_matrices(
        args.adjacency_path, args.distance_path, device
    )
    
    # Load trip tensor
    temporal_adjacency_tensor = load_trip_tensor(args.trips_tensor_path, device)
    
    # Data split
    split_idx = int(args.train_split * len(temporal_adjacency_tensor))
    train_tensor = temporal_adjacency_tensor[:split_idx].to(device)
    test_tensor = temporal_adjacency_tensor[split_idx:].to(device)
    
    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Test tensor shape: {test_tensor.shape}")
    
    # Model configuration
    N = train_tensor.shape[1]  # Number of nodes
    
    if args.multi_step:
        num_prediction_horizons = len(args.prediction_horizons)
    else:
        num_prediction_horizons = 1
    
    model_config = {
        'N': N,
        'in_features': N,
        'hidden_size': args.hidden_size,
        'W_long': args.w_long,
        'chunk_size_short': args.chunk_size_short,
        'num_chunks_short': args.num_chunks_short,
        'num_prediction_horizons': num_prediction_horizons,
        'adj_matrix_tensor': adj_matrix_tensor,
        'dist_matrix_tensor': dist_matrix_tensor
    }
    
    # Load model
    model = load_model(args.checkpoint_path, model_config, device)
    
    # Evaluate on training set
    if args.eval_train:
        evaluate_dataset(model, train_tensor, "Training", args, device, output_dir)
    
    # Evaluate on test set
    if args.eval_test:
        evaluate_dataset(model, test_tensor, "Test", args, device, output_dir)
    
    # Perform recursive forecasting
    if args.recursive_forecast:
        perform_recursive_forecasting(model, test_tensor, args, device, output_dir)
    
    print("\nEvaluation script completed successfully!")


if __name__ == "__main__":
    main()