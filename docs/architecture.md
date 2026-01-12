# Model Architecture

## Overview

The FusedODModel is a deep neural network designed for spatiotemporal Origin-Destination (OD) flow prediction. It combines spatial learning through Graph Attention Networks (GAT) with temporal learning through recurrent neural networks.

## Architecture Components

### 1. Graph Attention Layer (GAT)

The GAT component captures spatial relationships between locations:

- **Input**: Node features for all locations at each time step
- **Adjacency Matrix**: Defines spatial connectivity between locations  
- **Distance Matrix**: Provides distance-based attention weighting
- **Output**: Spatially-aware node embeddings

Key features:
- Distance-aware attention mechanism
- Learnable distance scaling parameter
- Xavier uniform weight initialization

### 2. Temporal Processing Branches

#### Long-term Branch
- **LSTM**: Processes the full input window (W_long steps)
- **Temporal Attention**: Weights LSTM hidden states to create context vector
- **Purpose**: Captures long-term temporal dependencies

#### Short-term Branch  
- **Multiple GRUs**: Each processes a short chunk of recent time steps
- **Individual Temporal Attentions**: One per GRU for focused context extraction
- **Purpose**: Captures recent fine-grained temporal patterns

### 3. Fusion and Decoding

- **Fusion Layer**: Concatenates context vectors from both branches
- **Dense Layers**: Transform fused representation
- **Conv2D**: Applies spatial smoothing to output matrices
- **Softplus**: Ensures non-negative predictions

## Multi-step Prediction

For multi-horizon forecasting:
- Output layer generates predictions for multiple future time steps
- Each horizon processed independently through Conv2D
- Supports arbitrary prediction horizons (1, 6, 12, 24 hours, etc.)

## Mathematical Formulation

### GAT Attention
```
e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) - α * log(d_ij + ε)
α_ij = softmax(e_ij)
h'_i = σ(Σ_j α_ij * Wh_j)
```

Where:
- `W`: Learned transformation matrix
- `a`: Attention mechanism weights
- `d_ij`: Distance between nodes i and j
- `α`: Learnable distance scaling parameter

### Temporal Attention
```
e_t = tanh(h_t * w_att)
β_t = softmax(e_t)
c = Σ_t β_t * h_t
```

Where:
- `h_t`: Hidden state at time t
- `w_att`: Learned attention weights
- `c`: Context vector

## Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| N | Number of spatial nodes | Data-dependent |
| W_long | Long-term window length | 144 |
| W_short | Short-term window length | 36 |
| chunk_size_short | Short-term chunk size | 9 |
| num_chunks_short | Number of short chunks | 4 |
| hidden_size | Hidden layer dimension | 64 |
| num_prediction_horizons | Prediction horizons | 1 |

## Loss Functions

### Negative Binomial NLL
For overdispersed count data:
```
L = -Σ [log Γ(y + r) - log Γ(y + 1) - log Γ(r) + r*log(r/(r+μ)) + y*log(μ/(r+μ))]
```

### Poisson NLL  
For standard count data:
```
L = -Σ [y*log(μ) - μ - log Γ(y + 1)]
```

Where:
- `y`: True counts
- `μ`: Predicted means
- `r`: Dispersion parameter (learnable)