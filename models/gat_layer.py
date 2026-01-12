"""
Graph Attention Layer implementation for spatial learning.

Based on Veličković et al., 2018 (https://arxiv.org/abs/1710.10903)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Basic single-head Graph Attention Layer.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension  
        dropout: Dropout probability
        alpha: Negative slope of LeakyReLU
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # Negative slope of LeakyReLU

        # Transformation weights
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Learnable scalar to weigh the influence of distance
        self.distance_scaler = nn.Parameter(torch.ones(1))

    def forward(self, h: torch.Tensor, adj: torch.Tensor, dist: torch.Tensor):
        """
        Forward pass of GAT layer.
        
        Args:
            h: Input features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            dist: Distance matrix (num_nodes, num_nodes)
            
        Returns:
            Output features (batch_size, num_nodes, out_features)
        """
        # Linear transformation
        h_prime = torch.einsum("bnf,fh->bnh", h, self.W)

        # Prepare for attention computation
        h_prime_i_repeat = h_prime.unsqueeze(2).repeat(1, 1, h_prime.shape[1], 1)
        h_prime_j_repeat = h_prime.unsqueeze(1).repeat(1, h_prime.shape[1], 1, 1)

        # Concatenate features for attention
        e_input = torch.cat((h_prime_i_repeat, h_prime_j_repeat), dim=3)

        # Feature-based attention scores
        e_feature_based = self.leakyrelu(torch.einsum("bnij,jk->bnik", e_input, self.a)).squeeze(3)

        # Distance-based attention influence
        dist_expanded = dist.unsqueeze(0).expand_as(e_feature_based)
        # Use negative log distance, scaled by learnable parameter. Add epsilon to prevent log(0).
        distance_influence = -torch.log(dist_expanded + 1e-6) * self.distance_scaler

        # Combine feature-based attention with distance influence
        combined_attention_scores = e_feature_based + distance_influence

        # Apply adjacency mask
        zero_vec = -9e15 * torch.ones_like(combined_attention_scores)
        adj_expanded = adj.unsqueeze(0).expand_as(combined_attention_scores)
        attention = torch.where(adj_expanded > 0, combined_attention_scores, zero_vec)

        # Softmax and dropout
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to get final output
        h_prime_aggregated = torch.einsum("bij,bjh->bih", attention, h_prime)

        return F.elu(h_prime_aggregated)