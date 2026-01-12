"""
Training utilities for FusedODModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.optim.lr_scheduler

from .data_loaders import build_windows_generator_single_step_fused, build_windows_generator_multi_step_fused
from ..models.losses import NegativeBinomialNLLLoss


def train_model_generator_fused(model, series_OD: torch.Tensor, W_long: int, W_short: int, 
                               lr: float, epochs: int, bs: int, ckpt_path: str, device: torch.device, 
                               patience: int = 10, min_delta: float = 1e-4, use_poisson_loss: bool = False):
    """
    Train FusedODModel for single-step prediction.
    
    Args:
        model: FusedODModel instance
        series_OD: Training data tensor (T, N, N)
        W_long: Long-term window length
        W_short: Short-term window length
        lr: Learning rate
        epochs: Number of training epochs
        bs: Batch size
        ckpt_path: Checkpoint save path
        device: Training device
        patience: Early stopping patience
        min_delta: Minimum improvement delta
        use_poisson_loss: Whether to use Poisson loss instead of Negative Binomial
        
    Returns:
        losses: List of training losses per epoch
    """
    # Initialize dispersion parameter for Negative Binomial loss
    r_param = None
    if not use_poisson_loss:
        r_param = nn.Parameter(torch.tensor(1.0, device=device))
        optimizer = torch.optim.Adam(list(model.parameters()) + [r_param], lr=lr)
        loss_fn = NegativeBinomialNLLLoss()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.PoissonNLLLoss(log_input=False)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float("inf")
    losses = []

    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        count = 0
        data_generator = build_windows_generator_single_step_fused(series_OD, W_long, W_short, bs)

        for xb_long, xb_short, yb in data_generator:
            xb_long, yb = xb_long.to(device), yb.to(device)
            optimizer.zero_grad()
            
            with autocast():
                pred = model(xb_long)  # FusedODModel takes only xb_long
                pred = torch.clamp(pred, min=1e-8)  # Ensure mu is positive

                if not use_poisson_loss:
                    r_positive = F.softplus(r_param)
                    loss = loss_fn(pred, yb, r_positive)
                else:
                    loss = loss_fn(pred, yb)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            count += 1

        avg = total / count if count > 0 else 0
        losses.append(avg)
        
        # Print progress
        if not use_poisson_loss:
            print(f"Ep {ep:02d} — Train NB NLL Loss: {avg:.6f}, r_param: {F.softplus(r_param).item():.4f}")
        else:
            print(f"Ep {ep:02d} — Train Poisson NLL Loss: {avg:.6f}")

        scheduler.step(avg)

        # Save best model
        if avg < best_loss:
            best_loss = avg
            save_dict = {'model_state_dict': model.state_dict()}
            if not use_poisson_loss:
                save_dict['r_param'] = r_param.item()
            torch.save(save_dict, ckpt_path)

        # Save periodic checkpoints
        if ep % 10 == 0:
            periodic_ckpt_path = ckpt_path.replace(".pth", f"_epoch_{ep}.pth")
            save_dict = {'model_state_dict': model.state_dict()}
            if not use_poisson_loss:
                save_dict['r_param'] = r_param.item()
            torch.save(save_dict, periodic_ckpt_path)
            print(f"Saved model checkpoint at epoch {ep} to {periodic_ckpt_path}")

    return losses


def train_model_generator_multi_step_fused(model, series_OD: torch.Tensor, W_long: int, prediction_horizons: list,
                                          lr: float, epochs: int, bs: int, ckpt_path: str, device: torch.device,
                                          patience: int = 10, min_delta: float = 1e-4, use_poisson_loss: bool = False):
    """
    Train FusedODModel for multi-step prediction.
    
    Args:
        model: FusedODModel instance
        series_OD: Training data tensor (T, N, N)
        W_long: Long-term window length
        prediction_horizons: List of prediction horizons
        lr: Learning rate
        epochs: Number of training epochs
        bs: Batch size
        ckpt_path: Checkpoint save path
        device: Training device
        patience: Early stopping patience
        min_delta: Minimum improvement delta
        use_poisson_loss: Whether to use Poisson loss instead of Negative Binomial
        
    Returns:
        losses: List of training losses per epoch
    """
    # Initialize dispersion parameter for Negative Binomial loss
    r_param = None
    if not use_poisson_loss:
        r_param = nn.Parameter(torch.tensor(1.0, device=device))
        optimizer = torch.optim.Adam(list(model.parameters()) + [r_param], lr=lr)
        loss_fn = NegativeBinomialNLLLoss()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.PoissonNLLLoss(log_input=False)

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float("inf")
    losses = []

    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        count = 0
        data_generator = build_windows_generator_multi_step_fused(series_OD, W_long, prediction_horizons, bs)

        for xb_long, yb_multi_step in data_generator:
            xb_long, yb_multi_step = xb_long.to(device), yb_multi_step.to(device)
            optimizer.zero_grad()
            
            with autocast():
                pred_multi_step = model(xb_long)
                pred_multi_step = torch.clamp(pred_multi_step, min=1e-8)

                if not use_poisson_loss:
                    r_positive = F.softplus(r_param)
                    loss = loss_fn(pred_multi_step, yb_multi_step, r_positive)
                else:
                    loss = loss_fn(pred_multi_step, yb_multi_step)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            count += 1

        avg = total / count if count > 0 else 0
        losses.append(avg)

        # Print progress
        if not use_poisson_loss:
            print(f"Ep {ep:02d} — Train NB NLL Loss: {avg:.6f}, r_param: {F.softplus(r_param).item():.4f}")
        else:
            print(f"Ep {ep:02d} — Train Poisson NLL Loss: {avg:.6f}")

        scheduler.step(avg)

        # Save best model
        if avg < best_loss:
            best_loss = avg
            save_dict = {'model_state_dict': model.state_dict()}
            if not use_poisson_loss:
                save_dict['r_param'] = r_param.item()
            torch.save(save_dict, ckpt_path)
            print(f"New best model saved with loss: {best_loss:.6f}")

        # Save periodic checkpoints
        if ep % 10 == 0:
            periodic_ckpt_path = ckpt_path.replace(".pth", f"_epoch_{ep}.pth")
            save_dict = {'model_state_dict': model.state_dict()}
            if not use_poisson_loss:
                save_dict['r_param'] = r_param.item()
            torch.save(save_dict, periodic_ckpt_path)
            print(f"Saved model checkpoint at epoch {ep} to {periodic_ckpt_path}")

    return losses