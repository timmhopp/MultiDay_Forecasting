#!/bin/bash

# Example evaluation script for FusedODModel

# Set paths (adjust these to your data locations)
ADJACENCY_PATH="/path/to/adjacency_matrix.csv"
DISTANCE_PATH="/path/to/distance_matrix.csv"
TRIPS_TENSOR_PATH="/path/to/trips_tensor.pt"
CHECKPOINT_PATH="/path/to/trained_model.pth"
OUTPUT_DIR="./evaluation_outputs"

# Evaluation parameters
BATCH_SIZE=64
HIDDEN_SIZE=64

# Window parameters (must match training)
W_LONG=144
W_SHORT=36
CHUNK_SIZE_SHORT=9
NUM_CHUNKS_SHORT=4

# Forecasting parameters
FORECAST_STEPS=144
PLOT_HORIZONS="1 6 18 72 144"

echo "Starting FusedODModel evaluation..."

# Standard evaluation
python eval_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT \
    --eval-train \
    --eval-test

echo "Basic evaluation completed!"

# Recursive forecasting evaluation
echo "Starting recursive forecasting..."

python eval_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir "${OUTPUT_DIR}_forecast" \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT \
    --recursive-forecast \
    --forecast-steps $FORECAST_STEPS \
    --plot-horizons $PLOT_HORIZONS

echo "Recursive forecasting completed!"

# For multi-step model evaluation, add --multi-step flag:
# python eval_main.py \
#     --adjacency-path $ADJACENCY_PATH \
#     --distance-path $DISTANCE_PATH \
#     --trips-tensor-path $TRIPS_TENSOR_PATH \
#     --checkpoint-path $CHECKPOINT_PATH \
#     --multi-step \
#     --prediction-horizons 1 36 144 432 1008 \
#     --output-dir "${OUTPUT_DIR}_multi_step" \
#     --eval-test \
#     --recursive-forecast