#!/bin/bash

# Example training script for FusedODModel

# Set paths (adjust these to your data locations)
ADJACENCY_PATH="/path/to/adjacency_matrix.csv"
DISTANCE_PATH="/path/to/distance_matrix.csv" 
TRIPS_TENSOR_PATH="/path/to/trips_tensor.pt"
CHECKPOINT_PATH="/path/to/model_checkpoint.pth"
OUTPUT_DIR="./training_outputs"

# Training parameters
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-3
HIDDEN_SIZE=64

# Window parameters
W_LONG=144
W_SHORT=36
CHUNK_SIZE_SHORT=9
NUM_CHUNKS_SHORT=4

echo "Starting FusedODModel training..."

# Single-step training
python train_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT

echo "Training completed!"

# Uncomment for multi-step training:
# python train_main.py \
#     --adjacency-path $ADJACENCY_PATH \
#     --distance-path $DISTANCE_PATH \
#     --trips-tensor-path $TRIPS_TENSOR_PATH \
#     --checkpoint-path "${CHECKPOINT_PATH%.pth}_multi_step.pth" \
#     --output-dir "${OUTPUT_DIR}_multi_step" \
#     --multi-step \
#     --prediction-horizons 1 36 144 432 1008 \
#     --epochs $EPOCHS \
#     --batch-size $BATCH_SIZE \
#     --lr $LEARNING_RATE