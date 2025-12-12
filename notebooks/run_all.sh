#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=0:10:00
#PBS -l select=1:ncpus=32:mem=256gb:ngpus=1:gpu_mem=95830mb:gpu_cap=sm_90:cl_bee=True
#PBS -N Fusion_train
#PBS -W group_list=cvut_fbmi_kbi
#PBS -W umask=002

DATASET_DIR="/mnt/e/Data/Fuse"

python model_original_train.py \
    --epochs 20\
    --train-path "$DATASET_DIR/fusionai_train_sim.txt" \
    --train-target "$DATASET_DIR/fusionai_train_target.csv" \
    --test-path "$DATASET_DIR/fusionai_test_sim.txt" \
    --test-target "$DATASET_DIR/fusionai_test_target.csv" \
    --output-folder "models_output" \
    
    