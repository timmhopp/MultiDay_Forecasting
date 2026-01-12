"""
Example script showing how to use FusedODModel for basic training and evaluation.

This script demonstrates:
1. Loading data
2. Setting up the model
3. Training the model
4. Evaluating performance
5. Making predictions
"""

import os
import torch
import numpy as np
from pathlib import Path

# Import FusedODModel components
from models import FusedODModel
from utils import load_adjacency_distance_matrices, load_trip_tensor, compute_metrics
from training import train_model_generator_fused
from evaluation import eval_fused_model


def main():
    """Run basic example."""
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths (update these to your actual data paths)
    adjacency_path = "path/to/adjacency_matrix.csv"
    distance_path = "path/to/distance_matrix.csv" 
    trips_tensor_path = "path/to/trips_tensor.pt"
    
    # Check if files exist (for demo purposes, we'll create dummy data if not)
    if not all(os.path.exists(p) for p in [adjacency_path, distance_path, trips_tensor_path]):
        print("Data files not found. Creating dummy data for demonstration...")
        N = 10  # Number of nodes
        T = 1000  # Number of time steps
        
        # Create dummy adjacency matrix (fully connected)
        adj_matrix = torch.ones(N, N)
        
        # Create dummy distance matrix (random distances)
        dist_matrix = torch.rand(N, N) + 0.1  # Add 0.1 to avoid zeros
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Make symmetric
        torch.fill_diagonal_(dist_matrix, 0.1)  # Small diagonal values
        
        # Create dummy trip tensor (random count data)
        trip_tensor = torch.randint(0, 50, (T, N, N)).float()
        
    else:
        # Load real data
        print("Loading data...")
        adj_matrix, dist_matrix = load_adjacency_distance_matrices(
            adjacency_path, distance_path, device
        )
        trip_tensor = load_trip_tensor(trips_tensor_path, device)
        N = adj_matrix.shape[0]
        T = trip_tensor.shape[0]
    
    print(f"Data loaded: {T} time steps, {N} locations")
    
    # Split data
    split_idx = int(0.8 * T)
    train_tensor = trip_tensor[:split_idx].to(device)
    test_tensor = trip_tensor[split_idx:].to(device)
    
    print(f"Train: {train_tensor.shape[0]} steps, Test: {test_tensor.shape[0]} steps")
    
    # Model configuration
    model_config = {
        'N': N,
        'in_features': N,
        'hidden_size': 32,  # Smaller for demo
        'W_long': 24,  # Shorter for demo
        'chunk_size_short': 6,
        'num_chunks_short': 2,
        'num_prediction_horizons': 1,
        'adj_matrix_tensor': adj_matrix,
        'dist_matrix_tensor': dist_matrix
    }
    
    # Initialize model
    model = FusedODModel(**model_config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training parameters
    training_config = {
        'lr': 1e-3,
        'epochs': 10,  # Few epochs for demo
        'bs': 16,
        'ckpt_path': 'demo_model.pth',
        'device': device,
        'use_poisson_loss': True  # Simpler loss for demo
    }
    
    # Train model
    print("\nTraining model...")
    losses = train_model_generator_fused(
        model=model,
        series_OD=train_tensor,
        W_long=model_config['W_long'],
        W_short=model_config['chunk_size_short'] * model_config['num_chunks_short'],
        **training_config
    )
    
    print(f"Training completed. Final loss: {losses[-1]:.6f}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    y_true, y_pred, metrics = eval_fused_model(
        model=model,
        series_OD=test_tensor,
        W_long=model_config['W_long'],
        W_short=model_config['chunk_size_short'] * model_config['num_chunks_short'],
        bs=training_config['bs'],
        device=device
    )
    
    print("\nTest Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Make a single prediction example
    print("\nMaking single prediction example...")
    model.eval()
    with torch.no_grad():
        # Take the first available test window
        W_long = model_config['W_long']
        if len(test_tensor) > W_long:
            input_window = test_tensor[:W_long].unsqueeze(0)  # Add batch dimension
            prediction = model(input_window)
            true_next = test_tensor[W_long]
            
            print(f"Input window shape: {input_window.shape}")
            print(f"Prediction shape: {prediction.shape}")
            print(f"True next step shape: {true_next.shape}")
            print(f"Prediction sum: {prediction.sum().item():.2f}")
            print(f"True sum: {true_next.sum().item():.2f}")
    
    print("\nExample completed successfully!")
    
    # Clean up demo files
    if os.path.exists('demo_model.pth'):
        os.remove('demo_model.pth')
        print("Demo checkpoint file cleaned up.")


if __name__ == "__main__":
    main()