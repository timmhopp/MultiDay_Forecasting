#!/usr/bin/env python3
"""
Main training script for FusedODModel.

This script handles training of the spatiotemporal OD (Origin-Destination) 
prediction model with both single-step and multi-step capabilities.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Import model components
from models.fused_od_model import FusedODModel
from models.gat_layer import GraphAttentionLayer
from models.temporal_attention import TemporalAttention
from models.losses import NegativeBinomialNLLLoss

# Import training utilities
from training.trainer import train_model_generator_fused, train_model_generator_multi_step_fused
from training.data_loaders import build_windows_generator_single_step_fused, build_windows_generator_multi_step_fused

# Import evaluation and visualization
from evaluation.evaluator import eval_fused_model, eval_fused_model_multi_step
from utils.visualization import plot_loss_curve
from utils.metrics import compute_metrics
from utils.data_utils import load_adjacency_distance_matrices, load_trip_tensor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FusedODModel for OD prediction")
    
    # Data paths
    parser.add_argument("--adjacency-path", type=str, required=True,
                       help="Path to adjacency matrix CSV file")
    parser.add_argument("--distance-path", type=str, required=True,
                       help="Path to distance matrix CSV file")
    parser.add_argument("--trips-tensor-path", type=str, required=True,
                       help="Path to trips tensor PT file")
    
    # Model parameters
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
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Train/test split ratio")
    
    # Multi-step prediction
    parser.add_argument("--multi-step", action="store_true",
                       help="Enable multi-step prediction training")
    parser.add_argument("--prediction-horizons", nargs="+", type=int, 
                       default=[1, 36, 144, 432, 1008],
                       help="Prediction horizons for multi-step training")
    
    # Loss function
    parser.add_argument("--use-poisson-loss", action="store_true",
                       help="Use Poisson loss instead of Negative Binomial")
    
    # Output paths
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to save model checkpoints")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Directory to save outputs")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup compute device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    # Check file existence
    if not os.path.exists(args.adjacency_path):
        print(f"Error: Adjacency file not found at {args.adjacency_path}")
        sys.exit(1)
    if not os.path.exists(args.distance_path):
        print(f"Error: Distance file not found at {args.distance_path}")
        sys.exit(1)
    if not os.path.exists(args.trips_tensor_path):
        print(f"Error: Trips tensor file not found at {args.trips_tensor_path}")
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
    
    # Model parameters
    N = train_tensor.shape[1]  # Number of nodes
    
    # Setup prediction parameters
    if args.multi_step:
        num_prediction_horizons = len(args.prediction_horizons)
        print(f"Multi-step training with horizons: {args.prediction_horizons}")
    else:
        num_prediction_horizons = 1
        args.prediction_horizons = [1]
        print("Single-step training")
    
    # Initialize model
    print("Initializing model...")
    model = FusedODModel(
        N=N,
        in_features=N,
        hidden_size=args.hidden_size,
        W_long=args.w_long,
        chunk_size_short=args.chunk_size_short,
        num_chunks_short=args.num_chunks_short,
        num_prediction_horizons=num_prediction_horizons,
        adj_matrix_tensor=adj_matrix_tensor,
        dist_matrix_tensor=dist_matrix_tensor
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training
    if args.epochs > 0:
        print("Starting training...")
        
        # Check if checkpoint exists
        if os.path.exists(args.checkpoint_path):
            print(f"Loading existing checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Train model
        if args.multi_step:
            losses = train_model_generator_multi_step_fused(
                model=model,
                series_OD=train_tensor,
                W_long=args.w_long,
                prediction_horizons=args.prediction_horizons,
                lr=args.lr,
                epochs=args.epochs,
                bs=args.batch_size,
                ckpt_path=args.checkpoint_path,
                device=device,
                use_poisson_loss=args.use_poisson_loss
            )
        else:
            losses = train_model_generator_fused(
                model=model,
                series_OD=train_tensor,
                W_long=args.w_long,
                W_short=args.w_short,
                lr=args.lr,
                epochs=args.epochs,
                bs=args.batch_size,
                ckpt_path=args.checkpoint_path,
                device=device,
                use_poisson_loss=args.use_poisson_loss
            )
        
        # Plot and save loss curve
        loss_plot_path = output_dir / "training_loss.png"
        plot_loss_curve(losses, "Training Loss", save_path=str(loss_plot_path))
        
        # Save losses
        loss_data_path = output_dir / "training_losses.npz"
        np.savez(loss_data_path, losses=losses)
        
        print(f"Training completed. Loss curve saved to {loss_plot_path}")
        print(f"Loss data saved to {loss_data_path}")
    
    else:
        print("Skipping training (epochs=0)")
        if os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluation on training set
    print("\nEvaluating on training set...")
    if args.multi_step:
        y_true_train, y_pred_train, metrics_train = eval_fused_model_multi_step(
            model, train_tensor, args.w_long, args.prediction_horizons, args.batch_size, device
        )
        print("Training set metrics (multi-step):")
        for horizon_key, metrics in metrics_train.items():
            print(f"  {horizon_key}:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")
    else:
        y_true_train, y_pred_train, metrics_train = eval_fused_model(
            model, train_tensor, args.w_long, args.w_short, args.batch_size, device
        )
        print("Training set metrics:")
        for k, v in metrics_train.items():
            print(f"  {k}: {v:.4f}")
    
    # Save training evaluation results
    train_results_path = output_dir / "train_evaluation_results.npz"
    if args.multi_step:
        save_data = {
            'y_true_by_horizon': y_true_train,
            'y_pred_by_horizon': y_pred_train,
            'prediction_horizons': args.prediction_horizons,
            'metrics': metrics_train
        }
    else:
        save_data = {
            'y_true': y_true_train,
            'y_pred': y_pred_train,
            'metrics': metrics_train
        }
    np.savez(train_results_path, **save_data)
    print(f"Training evaluation results saved to {train_results_path}")
    
    print("Training script completed successfully!")


if __name__ == "__main__":
    main()