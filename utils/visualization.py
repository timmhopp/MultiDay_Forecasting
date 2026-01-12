"""
Visualization utilities for OD prediction models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_od_heatmaps(y_true, y_pred, save_path=None):
    """
    Plot heatmaps comparing true vs predicted OD matrices.
    
    Args:
        y_true: True OD matrix
        y_pred: Predicted OD matrix
        save_path: Optional path to save the plot
    """
    err = y_true - y_pred
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    sns.heatmap(y_true, ax=axs[0], vmin=mn, vmax=mx, cbar=False)
    axs[0].set_title("True OD")
    
    sns.heatmap(y_pred, ax=axs[1], vmin=mn, vmax=mx, cbar=False)
    axs[1].set_title("Predicted OD")
    
    sns.heatmap(err, ax=axs[2], center=0, cbar=True)
    axs[2].set_title("Error")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_scatter(y_true, y_pred, save_path=None):
    """
    Plot scatter plot of true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    yt, yp = y_true.ravel(), y_pred.ravel()
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, alpha=0.3, s=10)
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.title("True vs Predicted")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_curve(losses, title, save_path=None):
    """
    Plot training loss curve.
    
    Args:
        losses: List of loss values
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(losses, marker='o', linewidth=2, markersize=4)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()