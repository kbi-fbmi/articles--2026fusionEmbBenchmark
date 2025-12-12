#!/bin/bash

dt=$(date '+%d_%m_%Y_%H_%M');
# Load Neccessary modules
#module load python/3.11.11

source .venv/bin/activate # add correct path corresponding to your virtual environment as step above


python ../extract_embd.py --path_data "../test_data/fusionai_test_sim.txt"--output_folder "./output" --output_name "evo_test"



