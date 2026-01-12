"""
Data utilities for loading and processing input data.
"""

import torch
import numpy as np
import pandas as pd
import os


def load_adjacency_distance_matrices(adjacency_path: str, distance_path: str, device: torch.device):
    """
    Load and process adjacency and distance matrices.
    
    Args:
        adjacency_path: Path to adjacency matrix CSV
        distance_path: Path to distance matrix CSV  
        device: Target device
        
    Returns:
        adj_matrix_tensor: Processed adjacency matrix tensor
        dist_matrix_tensor: Processed distance matrix tensor
    """
    # Load adjacency matrix
    adjacency = pd.read_csv(adjacency_path)
    if "Unnamed: 0" in adjacency.columns:
        adjacency.drop(columns=["Unnamed: 0"], inplace=True)
    adj_np = adjacency.to_numpy()
    np.fill_diagonal(adj_np, 1)  # Self-loops
    
    # Load distance matrix
    distance = pd.read_csv(distance_path)
    if "Unnamed: 0" in distance.columns:
        distance.drop(columns=["Unnamed: 0"], inplace=True)
    if "LocationID_pickup" in distance.columns:
        distance.drop(columns=["LocationID_pickup"], inplace=True)
    dist_np = distance.to_numpy()
    
    # Convert to tensors
    adj_matrix_tensor = torch.from_numpy(adj_np).float().to(device)
    dist_matrix_tensor = torch.from_numpy(dist_np).float().to(device)
    
    return adj_matrix_tensor, dist_matrix_tensor


def load_trip_tensor(trips_tensor_path: str, device: torch.device):
    """
    Load trip tensor data.
    
    Args:
        trips_tensor_path: Path to trips tensor PT file
        device: Target device
        
    Returns:
        temporal_adjacency_tensor: Loaded tensor data
    """
    loaded_data = torch.load(trips_tensor_path, weights_only=False, map_location=device)
    
    if isinstance(loaded_data, np.ndarray):
        temporal_adjacency_tensor = torch.from_numpy(loaded_data)
    elif isinstance(loaded_data, torch.Tensor):
        temporal_adjacency_tensor = loaded_data
    else:
        raise TypeError(f"Unexpected type loaded: {type(loaded_data)}. Expected numpy.ndarray or torch.Tensor.")
    
    return temporal_adjacency_tensor