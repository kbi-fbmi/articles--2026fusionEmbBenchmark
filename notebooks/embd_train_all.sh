#!/bin/bash
# filepath: run_all_embeddings_train.sh

# Configuration
DATA_FOLDER="download/dataset"
OUTPUT_FOLDER="download/results"
EPOCHS=300

# Sample sizes to test
SAMPLE_SIZES=(50 200 400 800 1200 2000 4000 8000 16000 24000 36302)

# Embedding types
EMBEDDING_TYPES=("nt" "evo" "bert" "hyena")

echo "Starting embedding training experiments..."
echo "Data folder: $DATA_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Epochs: $EPOCHS"
echo "Sample sizes: ${SAMPLE_SIZES[@]}"
echo "Embedding types: ${EMBEDDING_TYPES[@]}"

# Create main output folder
mkdir -p "$OUTPUT_FOLDER"

# Total number of experiments
TOTAL_EXPERIMENTS=$((${#EMBEDDING_TYPES[@]} * ${#SAMPLE_SIZES[@]}))
CURRENT_EXPERIMENT=0

echo ""
echo "Total experiments to run: $TOTAL_EXPERIMENTS"
echo "======================================================"

# Loop through each embedding type
for embedding in "${EMBEDDING_TYPES[@]}"; do
    echo ""
    echo "Training $embedding embeddings..."
    echo "--------------------------------"
    
    # Loop through each sample size
    for samples in "${SAMPLE_SIZES[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Training $embedding with $samples samples..."
        
        # Run the training
        python embd_train.py \
            --epochs $EPOCHS \
            --embedding-type $embedding \
            --data-folder "$DATA_FOLDER" \
            --output-folder "$OUTPUT_FOLDER" \
            --num-of-samples $samples
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $embedding with $samples samples"
        else
            echo "✗ Failed training $embedding with $samples samples"
        fi
        
        echo "Progress: $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS completed"
    done
    
    echo ""
    echo "Completed all sample sizes for $embedding"
done

echo ""
echo "======================================================"
echo "All embedding training experiments completed!"
echo "Results saved in: $OUTPUT_FOLDER"
echo ""

# List all generated files
echo "Generated files:"
ls -la "$OUTPUT_FOLDER"/*.keras | wc -l | xargs echo "Models:"
ls -la "$OUTPUT_FOLDER"/*.pkl | wc -l | xargs echo "History/Results files:"

echo ""
echo "To view results summary, check the individual result files in $OUTPUT_FOLDER"