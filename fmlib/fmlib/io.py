import json
import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
import zenodo_get as zg
from Bio import SeqIO
from tqdm import tqdm

from fmlib import fm


def zenodo_get_and_unzip(zenodo_id: str, download_file: str, destination_dir: str) -> None:
    """Download a dataset from Zenodo, unzip it, and organize it into the specified directory.

    Parameters
    ----------
    zenodo_id : str
        The Zenodo record ID to download.
    download_file : str
        The name of the zip file to download.
    destination_dir : str
        The directory where the dataset will be stored.

    Behavior
    --------
    - Check if the target dataset folder exists; if not, create it.
    - Download the dataset zip file from Zenodo using the provided ID.
    - Unzip the downloaded file into the destination directory.
    - Remove the zip file after extraction.
    - Print progress and error messages during the process.

    """
    dest_dir = Path(destination_dir)

    try:
        zg.download(zenodo_id, output_dir=str(dest_dir))
        print(f"Downloaded {dest_dir}.")

        # Unzip the downloaded file
        print(f"Unzipping {download_file}...")
        shutil.unpack_archive(dest_dir / download_file, destination_dir)
        print("Unzipping complete.")

        # Clean up the downloaded zip file
        Path.unlink(dest_dir / download_file)
        print(f"Removed {download_file}.")

    except Exception as e:
        print(f"Error downloading or unzipping the dataset: {e}")


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
