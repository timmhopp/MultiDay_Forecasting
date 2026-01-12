"""
Temporal Attention Layer for handling sequential dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Simple Temporal Attention Layer for weighting sequential inputs.
    
    Args:
        hidden_size: Size of the hidden dimension
    """
    
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        # Learnable weight vector for attention
        self.attn_weights = nn.Parameter(torch.rand(hidden_size, 1))
        
        # Query and Key transformations (optional)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        
        # Attention mechanism vector
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs):
        """
        Forward pass of temporal attention.
        
        Args:
            encoder_outputs: Sequential outputs (batch_size, seq_len, hidden_size)
            
        Returns:
            context_vector: Weighted context vector (batch_size, hidden_size)
        """
        # Compute attention scores using a simpler form
        # Energy: (batch_size, seq_len, 1)
        energy = torch.tanh(torch.matmul(encoder_outputs, self.attn_weights))
        
        # Attention weights: (batch_size, seq_len)
        attention_weights = F.softmax(energy.squeeze(2), dim=1)
        
        # Context vector: (batch_size, hidden_size)
        context_vector = torch.einsum("bs,bsd->bd", attention_weights, encoder_outputs)
        
        return context_vector