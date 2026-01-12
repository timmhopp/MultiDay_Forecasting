"""
Custom loss functions for OD prediction models.
"""

import torch
import torch.nn as nn


class NegativeBinomialNLLLoss(nn.Module):
    """
    Negative Binomial Negative Log-Likelihood Loss.
    
    This loss function is appropriate for count data with overdispersion,
    which is common in traffic flow prediction tasks.
    
    Args:
        eps: Small epsilon value to prevent numerical instabilities
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu, target, r):
        """
        Compute the negative log-likelihood loss.
        
        Args:
            mu: Predicted mean (from model output)
            target: True counts
            r: Dispersion parameter (learned or fixed)
            
        Returns:
            loss: Negative log-likelihood loss
        """
        # Ensure mu and r are positive
        mu = mu + self.eps
        r = r + self.eps

        # Compute log-gamma terms
        log_gamma_target_plus_r = torch.lgamma(target + r)
        log_gamma_target_plus_1 = torch.lgamma(target + 1)
        log_gamma_r = torch.lgamma(r)

        # Compute the three terms of the NB log-likelihood
        term1 = log_gamma_target_plus_r - log_gamma_target_plus_1 - log_gamma_r
        term2 = r * torch.log(r / (r + mu))
        term3 = target * torch.log(mu / (r + mu))

        # Return negative log-likelihood
        loss = -torch.sum(term1 + term2 + term3)
        return loss