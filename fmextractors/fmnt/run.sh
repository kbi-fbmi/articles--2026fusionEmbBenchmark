#!/bin/bash
echo "Running the script to extract embeddings..."
source ./.venv/bin/activate
# Run the Python script extract_embd.py extracting embeddings
python3 extract_embd_m.py --path_data "../../test_data/fusionai_test_sim.txt" --output_folder "./output" --output_name "nt_test"