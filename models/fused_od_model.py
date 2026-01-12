"""
FusedODModel: A spatiotemporal model for Origin-Destination prediction.

This model combines Graph Attention Networks (GAT) for spatial learning
with LSTM/GRU for temporal learning, using both long-term and short-term branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_layer import GraphAttentionLayer
from .temporal_attention import TemporalAttention


class FusedODModel(nn.Module):
    """
    Fused Origin-Destination prediction model combining spatial and temporal learning.
    
    Args:
        N: Number of spatial nodes/locations
        in_features: Input feature dimension 
        hidden_size: Hidden layer dimension
        W_long: Long-term input window length
        chunk_size_short: Size of each short-term chunk
        num_chunks_short: Number of short-term chunks
        num_prediction_horizons: Number of prediction horizons (1 for single-step)
        adj_matrix_tensor: Adjacency matrix tensor
        dist_matrix_tensor: Distance matrix tensor
    """
    
    def __init__(self, N, in_features, hidden_size=64, W_long=144, 
                 chunk_size_short=9, num_chunks_short=4, num_prediction_horizons=1,
                 adj_matrix_tensor: torch.Tensor = None, 
                 dist_matrix_tensor: torch.Tensor = None):
        super().__init__()
        self.N = N
        self.W_long = W_long  # Total input window length for long-term branch
        self.chunk_size_short = chunk_size_short  # Length of each short-term chunk
        self.num_chunks_short = num_chunks_short  # Number of short-term chunks
        self.W_short = self.chunk_size_short * self.num_chunks_short  # Total short window
        self.num_prediction_horizons = num_prediction_horizons

        # Shared GCN/GAT layer for both branches
        self.gcn = GraphAttentionLayer(in_features, hidden_size)

        # Long-term branch: single LSTM with temporal attention
        self.lstm_long = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.temporal_attention_long = TemporalAttention(hidden_size)
        self.register_buffer("feat_buffer_long", torch.empty(1, self.W_long, hidden_size))

        # Short-term branch: multiple GRUs with temporal attentions
        self.grus_short = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, batch_first=True) 
            for _ in range(self.num_chunks_short)
        ])
        self.temporal_attentions_short = nn.ModuleList([
            TemporalAttention(hidden_size) 
            for _ in range(self.num_chunks_short)
        ])
        self.register_buffer("feat_buffer_short", torch.empty(1, self.chunk_size_short, hidden_size))

        # Fusion layer
        fusion_input_dim = hidden_size + (hidden_size * self.num_chunks_short)
        self.linear1 = nn.Linear(fusion_input_dim, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 2, self.num_prediction_horizons * N * N)
        
        # Convolutional layer for spatial smoothing
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.softplus = nn.Softplus()

        # Register adjacency and distance matrices as buffers
        if adj_matrix_tensor is not None:
            self.register_buffer('adj_matrix_tensor', adj_matrix_tensor.float())
        else:
            raise ValueError("adj_matrix_tensor must be provided for FusedODModel with GAT.")

        if dist_matrix_tensor is not None:
            self.register_buffer('dist_matrix_tensor', dist_matrix_tensor.float())
        else:
            raise ValueError("dist_matrix_tensor must be provided for FusedODModel with GAT.")

    def forward(self, seg: torch.Tensor):
        """
        Forward pass of the model.
        
        Args:
            seg: Input segment (batch_size, W_long, N, N)
            
        Returns:
            out: Predictions (batch_size, num_prediction_horizons, N, N) or (batch_size, N, N) for single-step
        """
        B, total_W, _, _ = seg.size()

        # Ensure input matches expected window length
        if total_W != self.W_long:
            raise ValueError(f"Input segment length {total_W} does not match expected W_long {self.W_long}")

        # --- Long-term branch processing ---
        long_term_gcn_features = self.feat_buffer_long.expand(B, -1, -1).clone()
        for t in range(self.W_long):
            # Apply GCN to each time step
            H = self.gcn(seg[:, t, :, :], self.adj_matrix_tensor, self.dist_matrix_tensor)
            long_term_gcn_features[:, t] = H.mean(dim=1)  # Aggregate node features

        # Pass through LSTM and apply temporal attention
        lstm_output_long, _ = self.lstm_long(long_term_gcn_features)
        context_vector_long = self.temporal_attention_long(lstm_output_long)

        # --- Short-term branch processing ---
        short_term_context_vectors = []
        # Use the most recent W_short steps
        seg_short_input = seg[:, -self.W_short:, :, :]

        for i in range(self.num_chunks_short):
            start = i * self.chunk_size_short
            end = start + self.chunk_size_short
            chunk_short = seg_short_input[:, start:end, :, :]

            # Apply GCN to short-term chunk
            short_term_gcn_features = self.feat_buffer_short.expand(B, -1, -1).clone()
            for t in range(self.chunk_size_short):
                H = self.gcn(chunk_short[:, t, :, :], self.adj_matrix_tensor, self.dist_matrix_tensor)
                short_term_gcn_features[:, t] = H.mean(dim=1)

            # Pass through GRU and apply temporal attention
            gru_output_short, _ = self.grus_short[i](short_term_gcn_features)
            context_vector_short_i = self.temporal_attentions_short[i](gru_output_short)
            short_term_context_vectors.append(context_vector_short_i)

        # --- Fusion ---
        fused = torch.cat([context_vector_long] + short_term_context_vectors, dim=1)

        # Decoder operations
        out = self.relu1(self.linear1(fused))
        out = self.linear2(out)
        
        if self.num_prediction_horizons > 1:
            # Multi-step output: (B, num_prediction_horizons, N, N)
            out = out.view(B, self.num_prediction_horizons, self.N, self.N)
            
            # Apply Conv2d to each prediction horizon
            conv_input = out.view(B * self.num_prediction_horizons, 1, self.N, self.N)
            conv_output = self.conv2d(conv_input)
            out = conv_output.view(B, self.num_prediction_horizons, self.N, self.N)
        else:
            # Single-step output: (B, N, N)
            out = out.view(B, self.N, self.N)
            
            # Apply Conv2d
            conv_input = out.unsqueeze(1)
            conv_output = self.conv2d(conv_input)
            out = conv_output.squeeze(1)
        
        # Apply softplus for non-negative outputs
        out = self.softplus(out)

        return out