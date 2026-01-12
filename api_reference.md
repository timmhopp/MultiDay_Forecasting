# API Reference

## Main Classes

### FusedODModel

The main model class combining spatial and temporal learning.

```python
class FusedODModel(nn.Module):
    def __init__(self, N, in_features, hidden_size=64, W_long=144, 
                 chunk_size_short=9, num_chunks_short=4, 
                 num_prediction_horizons=1, adj_matrix_tensor=None, 
                 dist_matrix_tensor=None)
```

**Parameters:**
- `N` (int): Number of spatial nodes/locations
- `in_features` (int): Input feature dimension
- `hidden_size` (int): Hidden layer dimension
- `W_long` (int): Long-term input window length
- `chunk_size_short` (int): Size of each short-term chunk
- `num_chunks_short` (int): Number of short-term chunks
- `num_prediction_horizons` (int): Number of prediction horizons
- `adj_matrix_tensor` (torch.Tensor): Adjacency matrix
- `dist_matrix_tensor` (torch.Tensor): Distance matrix

**Methods:**
- `forward(seg)`: Forward pass through the model

### GraphAttentionLayer

Graph attention mechanism with distance-aware weighting.

```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2)
```

**Parameters:**
- `in_features` (int): Input feature dimension
- `out_features` (int): Output feature dimension
- `dropout` (float): Dropout probability
- `alpha` (float): LeakyReLU negative slope

### TemporalAttention

Temporal attention mechanism for sequence weighting.

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size)
```

**Parameters:**
- `hidden_size` (int): Hidden dimension size

## Training Functions

### train_model_generator_fused

Train model for single-step prediction.

```python
def train_model_generator_fused(model, series_OD, W_long, W_short, lr, 
                               epochs, bs, ckpt_path, device, 
                               patience=10, min_delta=1e-4, 
                               use_poisson_loss=False)
```

**Parameters:**
- `model`: FusedODModel instance
- `series_OD` (torch.Tensor): Training data (T, N, N)
- `W_long` (int): Long-term window length
- `W_short` (int): Short-term window length
- `lr` (float): Learning rate
- `epochs` (int): Number of training epochs
- `bs` (int): Batch size
- `ckpt_path` (str): Checkpoint save path
- `device` (torch.device): Training device
- `patience` (int): Early stopping patience
- `min_delta` (float): Minimum improvement delta
- `use_poisson_loss` (bool): Use Poisson instead of Negative Binomial loss

**Returns:**
- `losses` (list): Training losses per epoch

### train_model_generator_multi_step_fused

Train model for multi-step prediction.

```python
def train_model_generator_multi_step_fused(model, series_OD, W_long, 
                                          prediction_horizons, lr, epochs, 
                                          bs, ckpt_path, device, 
                                          patience=10, min_delta=1e-4, 
                                          use_poisson_loss=False)
```

Similar parameters to single-step version, with:
- `prediction_horizons` (list): List of prediction horizons

## Evaluation Functions

### eval_fused_model

Evaluate model for single-step prediction.

```python
def eval_fused_model(model, series_OD, W_long, W_short, bs, device)
```

**Returns:**
- `y_true` (np.ndarray): True values
- `y_pred` (np.ndarray): Predicted values  
- `metrics` (dict): Evaluation metrics

### eval_fused_model_multi_step

Evaluate model for multi-step prediction.

```python
def eval_fused_model_multi_step(model, series_OD, W_long, 
                               prediction_horizons, bs, device)
```

**Returns:**
- `y_true_by_horizon` (list): True values by horizon
- `y_pred_by_horizon` (list): Predicted values by horizon
- `metrics_by_horizon` (dict): Metrics by horizon

## Utility Functions

### compute_metrics

Calculate evaluation metrics.

```python
def compute_metrics(y_true, y_pred, eps=1e-6)
```

**Returns:**
- `metrics` (dict): Dictionary with MSE, RMSE, MAE, MAPE, SMAPE, R2

### load_adjacency_distance_matrices

Load and process spatial matrices.

```python
def load_adjacency_distance_matrices(adjacency_path, distance_path, device)
```

**Returns:**
- `adj_matrix_tensor` (torch.Tensor): Processed adjacency matrix
- `dist_matrix_tensor` (torch.Tensor): Processed distance matrix

### load_trip_tensor

Load trip tensor data.

```python
def load_trip_tensor(trips_tensor_path, device)
```

**Returns:**
- `tensor` (torch.Tensor): Loaded trip tensor

## Visualization Functions

### plot_od_heatmaps

Plot OD matrix comparison heatmaps.

```python
def plot_od_heatmaps(y_true, y_pred, save_path=None)
```

### plot_scatter

Plot true vs predicted scatter plot.

```python
def plot_scatter(y_true, y_pred, save_path=None)
```

### plot_loss_curve

Plot training loss curve.

```python
def plot_loss_curve(losses, title="Training Loss", save_path=None)
```