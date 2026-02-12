#!/bin/bash
echo "Running the script to extract embeddings..."
. ./.venv/bin/activate
# Run the Python script extract_embd.py extracting embeddings
python3 extract_embd.py --path_data "../../test_data/fusionai_test_sim.txt" --output_folder "./output" --output_name "bert_test_mean" --batch_size 2 --embd_type "mean"