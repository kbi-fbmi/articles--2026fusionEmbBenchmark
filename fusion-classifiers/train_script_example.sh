#!/bin/bash
echo "Running fusion ai training..."
source ./.venv/bin/activate

DATASET_DIR="../test_data"
OUTPUT_FOLDER="models_output"
EPOCHS=2

python train_fusionai.py \
    --epochs $EPOCHS\
    --train-path "$DATASET_DIR/fusionai_test_sim.txt" \
    --train-target "$DATASET_DIR/fusionai_test_target.csv" \
    --test-path "$DATASET_DIR/fusionai_test_sim.txt" \
    --test-target "$DATASET_DIR/fusionai_test_target.csv" \
    --output-folder "$OUTPUT_FOLDER" \
    

EMB_FOLDER="../notebooks/download/embeddings"
SAMPLES=100
EMBEDDING_TYPE="nt"  # Options: nt, evo, hyena, bert

python train_embeddings.py \
    --epochs $EPOCHS \
    --embedding-type $EMBEDDING_TYPE \
    --classifier-type "nn" \
    --data-folder "$EMB_FOLDER" \
    --output-folder "$OUTPUT_FOLDER" \
    --num-of-samples $SAMPLES