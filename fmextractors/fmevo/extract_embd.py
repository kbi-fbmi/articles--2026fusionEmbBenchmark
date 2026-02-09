import argparse
import concurrent.futures
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evo2 import Evo2
from fmlib import io
from fmlib.fm import (
    compute_metrics_summary,
    extr_key,
    save_metrics,
    tokenize_sequence_parallel,
)


def evo_tokenize(sequence):
    return list(np.frombuffer(sequence.encode("utf-8"), dtype=np.uint8))


def mock_evo_call(tokens, return_embeddings=False, layer_names=None):
    print(f"Mock Evo2 call with tokens: {tokens.size()} and layer_names: {layer_names}")
    return {"blocks.28.mlp.l3": torch.randn(tokens.size(0), tokens.size(1), 1024)}


def embeding_evo(
    tokens,
    batch_size,
    emb_positions,
    evo2_model,
    layer_name="blocks.28.mlp.l3",
    embd_type="middle",
):
    print(f"Starting embedding extraction for layer: {layer_name}")
    embeddings = torch.asarray([])
    batch_times = []

    # Track peak VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)

    for i in range(num_batches):
        batch_start = time.time()
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size]
        print(f"Processing batch {i + 1}/{num_batches} with {len(batch_tokens)} tokens")
        _, batch_embeddings = evo2_model(
            batch_tokens, return_embeddings=True, layer_names=[layer_name]
        )
        if embd_type == "mean":
            selected_embeddings = batch_embeddings[layer_name].mean(dim=1, keepdim=True)
        else:
            selected_embeddings = batch_embeddings[layer_name][:, emb_positions, :]
        embeddings = torch.cat((embeddings, selected_embeddings.cpu()), dim=0)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        print(f"Batch {i + 1} done - Time: {batch_time:.3f}s")

    metrics = {
        "batch_times": batch_times,
        "total_samples": len(tokens),
    }
    if torch.cuda.is_available():
        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("Embedding extraction completed")
    return embeddings, metrics


def main():
    parser = argparse.ArgumentParser(description="Process Evo2 embeddings.")
    parser.add_argument("--path_data", required=True, help="Path to the  data file")
    parser.add_argument(
        "--output_folder", required=True, help="Output folder for saving results"
    )
    parser.add_argument(
        "--output_name", required=True, help="Output prefix for saving results"
    )
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

    print(f"Loading training data from {PATH_DATA}")
    fusion_data = io.load_fusions_from_fusionaitxt(PATH_DATA)

    print("Initializing Evo2 model")
    evo2_model = Evo2("evo2_7b")

    print("Tokenizing sequences")
    nptokens_fusion1 = tokenize_sequence_parallel(
        extr_key(fusion_data, "sequence1"), evo_tokenize, 32
    )
    nptokens_fusion2 = tokenize_sequence_parallel(
        extr_key(fusion_data, "sequence2"), evo_tokenize, 32
    )

    tokens_fusion1 = torch.tensor(nptokens_fusion1, dtype=torch.int).to("cuda:0")
    tokens_fusion2 = torch.tensor(nptokens_fusion2, dtype=torch.int).to("cuda:0")

    emb_pos = [tokens_fusion1.size(1) // 2]
    print("Extracting embeddings for test sequences")

    total_start = time.time()

    emb1, metrics1 = embeding_evo(
        tokens_fusion1,
        4,
        emb_pos,
        evo2_model,
        layer_name="blocks.28.mlp.l3",
        embd_type=EMBD_TYPE,
    )
    emb2, metrics2 = embeding_evo(
        tokens_fusion2,
        4,
        emb_pos,
        evo2_model,
        layer_name="blocks.28.mlp.l3",
        embd_type=EMBD_TYPE,
    )

    total_time = time.time() - total_start

    print(f"Creating output folder at {OUTPUT_FOLDER}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CSV files")
    pd.DataFrame(emb1[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq1.csv", index=False, header=False
    )

    pd.DataFrame(emb2[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq2.csv", index=False, header=False
    )

    # Compute and save metrics using shared library
    metrics_summary = compute_metrics_summary(
        "Evo2", EMBD_TYPE, metrics1, metrics2, total_time
    )
    save_metrics(metrics_summary, OUTPUT_FOLDER, OUTPUT_NAME)

    print("\nProcessing completed successfully")


if __name__ == "__main__":
    main()
