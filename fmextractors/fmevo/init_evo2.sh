#!/bin/bash
# you can run this script to initialize the evo2 environment on a cluster. or ssh to the cluster and run it there.
dt=$(date '+%d_%m_%Y_%H_%M');
git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git
cd evo2

#  require python 3.11.11 and cuda 12.6

# set python enviroment
python -m venv .venv

# Load venv
source .venv/bin/activate

# Load Neccessary modules need to be modified according to your cluster or cuda 12.6 load
which python

python --version
nvcc --version 
which nvvc

cd vortex
pip install .
cd ..

ln -s $(pwd)/.venv/lib/python3.11/site-packages/nvidia/cudnn/include/cudnn* .venv/include
pip install .

#python main.py
pip install transformer_engine[pytorch]==1.13
pip install pandas
pip install -e ../../../fmlib
