import os

os.environ["KERAS_BACKEND"] = "torch"

import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from fmlib import fm


def read_fasta(file_path, fused_lambda=None):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        description = record.description
        seq_data = {
            "id": record.id,
            "sequence": str(record.seq),
            "name": str(record.name),
            "description": description,
            "fused_at": None,
        }

        if fused_lambda:
            try:
                seq_data["fused_at"] = fused_lambda(description)
            except ValueError:
                pass

        sequences.append(seq_data)
    return sequences


def real_extract_fp(input_string):
    return int(re.search(r"BP=(\d+)", input_string).group(1))


def sim_extract_fp(input_string):
    return int(re.search(r"FusedAt:(\d+)", input_string).group(1))


def save_fusions_to_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_fusions_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_fusions_from_fusionaitxt(file_path, fused_lambda=None):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            columns = line.strip().split("\t")
            if len(columns) == 11:
                entry = {
                    "gene1": columns[0],
                    "chr1": columns[1],
                    "pos1": int(columns[2]),
                    "strand1": columns[3],
                    "gene2": columns[4],
                    "chr2": columns[5],
                    "pos2": int(columns[6]),
                    "strand2": columns[7],
                    "sequence1": columns[8],
                    "sequence2": columns[9],
                    "target": columns[10],
                }
                if "N" in entry["sequence1"] or "N" in entry["sequence2"]:
                    continue
                data.append(entry)
    return data


def read_fasta_files(files, fused_lambda=None):
    sequences = []
    for f in files:
        sequences.extend(read_fasta(f, fused_lambda))
    return sequences


def hello():
    print("Hello from fbme biolib")


def convert_sequence_to_onehot_ACGT(sequence):
    """Convert a nucleotide sequence to a one-hot encoded representation (fast, numpy)."""
    mapping = np.zeros((256, 4), dtype=np.uint8)
    mapping[ord("A")] = np.array([1, 0, 0, 0], dtype=np.bool_)
    mapping[ord("C")] = np.array([0, 1, 0, 0], dtype=np.bool_)
    mapping[ord("G")] = np.array([0, 0, 1, 0], dtype=np.bool_)
    mapping[ord("T")] = np.array([0, 0, 0, 1], dtype=np.bool_)
    arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    try:
        onehot = mapping[arr]
    except IndexError:
        raise ValueError("Invalid nucleotide in sequence")
    if not np.all(onehot.sum(axis=1)):
        raise ValueError("Invalid nucleotide in sequence")
    return onehot


def load_fusion_embeddings(seq1_path, seq2_path, response_path):
    T1 = pd.read_csv(seq1_path, header=None)
    T2 = pd.read_csv(seq2_path, header=None)
    T1.columns = [f"{col}_T1" for col in T1.columns]
    T2.columns = [f"{col}_T2" for col in T2.columns]
    x = pd.concat([T1, T2], axis=1)

    # Load and encode labels
    y = pd.read_csv(response_path, header=None).iloc[:, 0]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Convert to torch tensors
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)

    return x_tensor, y_tensor
