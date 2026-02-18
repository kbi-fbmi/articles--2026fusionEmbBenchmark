#!/bin/bash

# Example script to test a saved model on custom data
#
# Usage: bash test_model.sh

# Set your paths here
MODEL_PATH="models_output/nt_mean_model_36302.keras"
SEQ1_PATH="../notebooks/download/embeddings/nt_test_seq1.csv"
SEQ2_PATH="../notebooks/download/embeddings/nt_test_seq2.csv"
TARGET_PATH="../notebooks/download/embeddings/fusionai_test_target.csv"
OUTPUT_PATH="test_results.pkl"  # Optional: comment out if you don't want to save results


# Run the test script
python test_saved_model.py \
    --model-path "$MODEL_PATH" \
    --seq1 "$SEQ1_PATH" \
    --seq2 "$SEQ2_PATH" \
    --target "$TARGET_PATH" \
    --output "$OUTPUT_PATH"

echo ""
echo "Testing complete!"
