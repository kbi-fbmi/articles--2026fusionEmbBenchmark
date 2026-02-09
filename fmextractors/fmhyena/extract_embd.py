import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from fmlib import io
from fmlib.fm import extr_key, compute_metrics_summary, save_metrics
from transformers import AutoModel, AutoTokenizer


# Model nastavenĂ­
MODEL_NAME = "LongSafari/hyenadna-large-1m-seqlen-hf"
MAX_LENGTH = 1000000

# Inicializace modelu a tokenizĂ©ru
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()


def embedding_hyena(tokens, batch_size, embd_type):
    projection_layer = nn.Linear(256, 2048).to(device)
    embeddings = []
    batch_times = []
    
    # Track peak VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)
    for i in range(num_batches):
        batch_start = time.time()
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size].to(device)

        with torch.no_grad():
            batch_embeddings = model(batch_tokens).last_hidden_state

        if embd_type == "mean":
            selected_embeddings = batch_embeddings.mean(dim=1, keepdim=True)
        else:
            seq_length = batch_embeddings.shape[1]
            middle_index = seq_length // 2
            selected_embeddings = batch_embeddings[:, middle_index : middle_index + 1, :]

        embeddings.append(selected_embeddings)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        print(f"Processing batch {i + 1}/{num_batches} - Saved {selected_embeddings.shape[0]} embeddings - Time: {batch_time:.3f}s")

    metrics = {
        "batch_times": batch_times,
        "total_samples": len(tokens),
    }
    if torch.cuda.is_available():
        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    
    return torch.cat(embeddings, dim=0), metrics


def save_embeddings(embeddings, output_folder, prefix):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving {prefix} embeddings to CSV files")
    pd.DataFrame(embeddings[:, 0, :].cpu().to(torch.float32).numpy()).to_csv(
        output_folder / f"{prefix}.csv", index=False, header=False
    )

    print(f"Saved {prefix} embeddings successfully")


def main():
    parser = argparse.ArgumentParser(description="Process Heyna embeddings.")
    parser.add_argument("--path_data", required=True, help="Path to the  data file")
    parser.add_argument("--output_folder", required=True, help="Output folder for saving results")
    parser.add_argument("--output_name", required=True, help="Output prefix for saving results")
    parser.add_argument(
        "--embd_type",
        default="middle",
        choices=["middle", "mean"],
        help="Type of embedding to extract: middle or mean",
    )
    args = parser.parse_args()

    PATH_DATA = args.path_data
    OUTPUT_FOLDER = args.output_folder
    OUTPUT_NAME = args.output_name
    EMBD_TYPE = args.embd_type

    fusion_data = io.load_fusions_from_fusionaitxt(PATH_DATA)
    
    # Track total processing time
    total_start = time.time()
    
    emb_seq1, metrics1 = embedding_hyena(
        tokenizer(
            extr_key(fusion_data, "sequence1"),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )["input_ids"].to(device),
        32,
        EMBD_TYPE,
    )

    emb_seq2, metrics2 = embedding_hyena(
        tokenizer(
            extr_key(fusion_data, "sequence2"),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )["input_ids"].to(device),
        32,
        EMBD_TYPE,
    )
    
    total_time = time.time() - total_start

    save_embeddings(emb_seq1, OUTPUT_FOLDER, f"{OUTPUT_NAME}_seq1")
    save_embeddings(emb_seq2, OUTPUT_FOLDER, f"{OUTPUT_NAME}_seq2")
    
    # Compute and save metrics using shared library
    metrics_summary = compute_metrics_summary(
        "HyenaDNA", EMBD_TYPE, metrics1, metrics2, total_time, device_info=torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
    )
    save_metrics(metrics_summary, OUTPUT_FOLDER, OUTPUT_NAME)
    
    print("\nProcessing completed successfully")


if __name__ == "__main__":
    main()
