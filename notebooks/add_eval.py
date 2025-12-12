import argparse
import os
import pickle as pkl
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"
import keras as kr
import pandas as pd
import torch

import ai

data_folder = Path("/mnt/e/Data/Fuse")
output_folder = Path("download/results")


train = ai.load_fusion_actg(data_folder / "fusionai_train_sim.txt", data_folder / "fusionai_train_target.csv")
test_full = ai.load_fusion_actg(data_folder / "fusionai_test_sim.txt", data_folder / "fusionai_test_target.csv")

test, valid = ai.split_test(test_full)
torch.manual_seed(42)

num_of_samples_all = [10, 200, 400, 800, 1200, 2000, 4000, 8000, 16000, 36302]

for num_of_samples in num_of_samples_all:
    model_path = Path(output_folder) / f"fusionai_model_{num_of_samples}.keras"
    model = kr.models.load_model(model_path)

    evaluation_results = ai.evaluate_model(model, test)
    with open(os.path.join(output_folder, f"fusionai_results_{num_of_samples}.pkl"), "wb") as f:
        pkl.dump(evaluation_results, f)
