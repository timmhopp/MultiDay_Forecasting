# FusedODModel: Spatiotemporal Origin-Destination Prediction

This project implements a deep learning model for predicting Origin-Destination (OD) flows using a combination of Graph Attention Networks (GAT) and recurrent neural networks (LSTM/GRU) on multiple temporal scales.

## Features

- **Spatial Learning**: Graph Attention Networks with distance-aware attention
- **Temporal Learning**: Long-term LSTM and short-term multi-GRU branches
- **Multi-step Prediction**: Predictions to single OD at specifc horizon
- **Flexible Loss Functions**: Negative Binomial and Poisson loss options
- **Recursive Forecasting**: Long-term recursive predictions (experimental)

## Project Structure

```
├── models/
│   ├── __init__.py
│   ├── fused_od_model.py      # Main FusedODModel implementation
│   ├── gat_layer.py           # Graph Attention Layer
│   ├── temporal_attention.py   # Temporal Attention mechanism
│   └── losses.py              # Custom loss functions
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training utilities
│   └── data_loaders.py        # Data loading and windowing
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           # Evaluation utilities
│   └── recursive_predictor.py # Recursive forecasting
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics
│   ├── data_utils.py          # Data processing utilities
│   └── visualization.py       # Plotting utilities
├── train_main.py              # Main training script
├── eval_main.py               # Main evaluation script
└── requirements.txt           # Python dependencies
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required data files:
   - Adjacency matrix CSV file
   - Distance matrix CSV file  
   - Trip tensor PT file

## Usage

### Training

```bash
python train_main.py \
    --adjacency-path /path/to/adjacency_matrix.csv \
    --distance-path /path/to/distance_matrix.csv \
    --trips-tensor-path /path/to/trips_tensor.pt \
    --checkpoint-path /path/to/model_checkpoint.pth \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3
```

For multi-step training:
```bash
python train_main.py \
    --adjacency-path /path/to/adjacency_matrix.csv \
    --distance-path /path/to/distance_matrix.csv \
    --trips-tensor-path /path/to/trips_tensor.pt \
    --checkpoint-path /path/to/model_checkpoint.pth \
    --multi-step \
    --prediction-horizons 1 36 144 432 1008 \
    --epochs 50
```

### Evaluation

```bash
python eval_main.py \
    --adjacency-path /path/to/adjacency_matrix.csv \
    --distance-path /path/to/distance_matrix.csv \
    --trips-tensor-path /path/to/trips_tensor.pt \
    --checkpoint-path /path/to/trained_model.pth \
    --output-dir ./evaluation_results
```

For recursive forecasting:
```bash
python eval_main.py \
    --adjacency-path /path/to/adjacency_matrix.csv \
    --distance-path /path/to/distance_matrix.csv \
    --trips-tensor-path /path/to/trips_tensor.pt \
    --checkpoint-path /path/to/trained_model.pth \
    --recursive-forecast \
    --forecast-steps 144 \
    --output-dir ./forecast_results
```

## Model Architecture

### FusedODModel

The main model combines:

1. **Graph Attention Layer**: Captures spatial relationships between locations with distance-aware attention
2. **Long-term Branch**: LSTM processing full input window with temporal attention
3. **Short-term Branch**: Multiple GRUs processing recent chunks with individual temporal attentions
4. **Fusion Layer**: Combines context vectors from both branches
5. **Decoder**: Generates OD matrix predictions with spatial smoothing

### Key Parameters

- `W_long`: Long-term input window length (default: 144)
- `W_short`: Short-term input window length (default: 36)
- `chunk_size_short`: Size of each short-term chunk (default: 9)
- `num_chunks_short`: Number of short-term chunks (default: 4)
- `hidden_size`: Hidden dimension size (default: 64)

## Data Format

### Input Files

1. **Adjacency Matrix CSV**: Square matrix indicating spatial connectivity
2. **Distance Matrix CSV**: Square matrix with pairwise distances between locations
3. **Trip Tensor PT**: PyTorch tensor with shape `(T, N, N)` where:
   - `T`: Number of time steps
   - `N`: Number of spatial locations
   - Values represent OD flows between locations

### Output Files

Training and evaluation generate:
- Model checkpoints (`.pth`)
- Evaluation results (`.npz`)
- Loss curves and visualizations (`.png`)

## Loss Functions

- **Negative Binomial NLL**: For overdispersed count data (default)
- **Poisson NLL**: For standard count data

## Evaluation Metrics

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

