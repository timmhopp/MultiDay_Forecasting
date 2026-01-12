"""
FusedODModel: Spatiotemporal Origin-Destination Flow Prediction

A deep learning framework combining Graph Attention Networks and 
recurrent neural networks for predicting origin-destination flows
in transportation networks.
"""

__version__ = "1.0.0"
__author__ = "FusedODModel Team"
__license__ = "MIT"

from .models import FusedODModel, GraphAttentionLayer, TemporalAttention, NegativeBinomialNLLLoss
from .utils import compute_metrics, load_adjacency_distance_matrices, load_trip_tensor

__all__ = [
    'FusedODModel',
    'GraphAttentionLayer', 
    'TemporalAttention',
    'NegativeBinomialNLLLoss',
    'compute_metrics',
    'load_adjacency_distance_matrices',
    'load_trip_tensor'
]