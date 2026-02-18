#!/bin/bash

# Configuration
EMB_FOLDER="../notebooks/download/embeddings_augmented"
MODEL_PATH="../notebooks/download/learned_models_nn/hyena_model_36302.keras"
#MODEL_PATH="models_output/hyena_mean_model_36302.keras"
OUTPUT_RESULTS="test_results"
TARGET_PATH="${EMB_FOLDER}/fusionai_test_target.csv"

# Create output directory
mkdir -p "$OUTPUT_RESULTS"

# Find all *_seq1.csv files and process them
for SEQ1_PATH in ${EMB_FOLDER}/hyena_s*_seq1.csv; do
    # Extract the base name (e.g., nt_mean_test from nt_mean_test_seq1.csv)
    BASENAME=$(basename "$SEQ1_PATH" _seq1.csv)
    
    # Construct corresponding seq2 path
    SEQ2_PATH="${EMB_FOLDER}/${BASENAME}_seq2.csv"
    
    # Check if seq2 file exists
    if [ ! -f "$SEQ2_PATH" ]; then
        echo "Warning: $SEQ2_PATH not found, skipping $BASENAME"
        continue
    fi
    
    # Extract embedding type for model matching (remove _test suffix if present)
    EMBEDDING_TYPE=${BASENAME%_test}
    
    # Find matching model file
   
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Warning: No model found for $EMBEDDING_TYPE, skipping"
        continue
    fi
    
    OUTPUT_FILE="${OUTPUT_RESULTS}/${BASENAME}_results.pkl"
    
    echo ""
    echo "=========================================="
    echo "Testing: $BASENAME"
    echo "Model: $MODEL_PATH"
    echo "=========================================="
    
    # Run the test script
    python test_saved_model.py \
        --model-path "$MODEL_PATH" \
        --seq1 "$SEQ1_PATH" \
        --seq2 "$SEQ2_PATH" \
        --target "$TARGET_PATH" \
        --output "$OUTPUT_FILE"
    
    echo "Results saved to: $OUTPUT_FILE"
done

echo ""
echo "All testing complete!"
