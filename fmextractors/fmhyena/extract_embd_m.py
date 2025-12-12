import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from fmlib import io

def extr_key(dict_list, key):
    print(f"Extracting key '{key}' from dictionary list")
    return [d[key] for d in dict_list]


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


# Model nastavenĂ­
MODEL_NAME = "LongSafari/hyenadna-large-1m-seqlen-hf"
MAX_LENGTH = 1000000

# Inicializace modelu a tokenizĂ©ru
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()


def embedding_hyena(tokens, batch_size):
    projection_layer = nn.Linear(256, 2048).to(device)
    embeddings = []

    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)
    for i in range(num_batches):
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size].to(device)

        with torch.no_grad():
            batch_embeddings = model(batch_tokens).last_hidden_state

        seq_length = batch_embeddings.shape[1]
        middle_index = seq_length // 2

        selected_embeddings = batch_embeddings[:, [0, middle_index, -1], :]

        embeddings.append(selected_embeddings)

        print(
            f"Processing batch {i + 1}/{num_batches} - Saved {selected_embeddings.shape[0]} embeddings"
        )

    return torch.cat(embeddings, dim=0)


def save_embeddings(embeddings, output_folder, prefix):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving {prefix} embeddings to CSV files")    
    pd.DataFrame(embeddings[:, 1, :].cpu().to(torch.float32).numpy()).to_csv(
        output_folder / f"{prefix}.csv", index=False, header=False
    )

    print(f"Saved {prefix} embeddings successfully")


def main():
    parser = argparse.ArgumentParser(description="Process Heyna embeddings.")
    parser.add_argument("--path_data", required=True, help="Path to the  data file")
    parser.add_argument("--output_folder", required=True, help="Output folder for saving results")
    parser.add_argument("--output_name", required=True, help="Output prefix for saving results")
    args = parser.parse_args()

    PATH_DATA = args.path_data
    OUTPUT_FOLDER = args.output_folder
    OUTPUT_NAME = args.output_name

    fusion_data = load_fusions_from_fusionaitxt(PATH_DATA)
    emb_seq1 = embedding_hyena(tokenizer(extr_key(fusion_data, "sequence1"),padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")["input_ids"].to(device), 32)
    save_embeddings(emb_seq1, OUTPUT_FOLDER, f"{OUTPUT_NAME}_seq1")
    emb_seq2 = embedding_hyena(tokenizer(extr_key(fusion_data, "sequence2"),padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")["input_ids"].to(device), 32)
    save_embeddings(emb_seq2, OUTPUT_FOLDER, f"{OUTPUT_NAME}_seq2")

    print("Processing completed successfully")


if __name__ == "__main__":
    main()
