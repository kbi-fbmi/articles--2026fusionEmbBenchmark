import argparse
import concurrent.futures
import time
from pathlib import Path
from typing import Any, Callable, List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from fmlib import io
from fmlib.fm import compute_metrics_summary, extr_key, save_metrics, tokenize_sequence_parallel
from jax.lib import xla_bridge
from nucleotide_transformer.pretrained import get_pretrained_model


def embeding_nt(
    tokens: jnp.ndarray,
    batch_size: int,
    emb_positions: List[int],
    nt_model: Any,
    layer_name: str,
    parameters: Any,
    random_key: Any,
    embd_type: str,
) -> tuple[np.ndarray, dict]:
    print(f"Starting embedding extraction for layer: {layer_name}")

    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)
    embeddings: np.ndarray | None = None
    batch_times = []

    for i in range(num_batches):
        batch_start = time.time()
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size]
        print(f"Processing batch {i + 1}/{num_batches} with {len(batch_tokens)} tokens")
        batch_embeddings = nt_model.apply(parameters, random_key, batch_tokens)
        if embd_type == "mean":
            selected_embeddings = np.mean(batch_embeddings[layer_name], axis=1, keepdims=True)
        else:
            selected_embeddings = batch_embeddings[layer_name][:, emb_positions, :]
        embeddings = (
            np.concatenate((embeddings, selected_embeddings), axis=0) if embeddings is not None else selected_embeddings
        )
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        print(f"Batch {i + 1} done - Time: {batch_time:.3f}s")

    metrics = {
        "batch_times": batch_times,
        "total_samples": len(tokens),
    }

    print("Embedding extraction completed")
    return embeddings, metrics


def main():
    # Configure logging

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process NT embeddings.")
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

    # PATH_DATA = "/mnt/e/Data/Fuse/fusionai_train_sim_107.txt"
    # OUTPUT_FOLDER = "./ouput"
    # OUTPUT_NAME = "nt_train"

    print(f"Loading training data from {PATH_DATA}")
    fusion_data = io.load_fusions_from_fusionaitxt(PATH_DATA)

    print("Initializing model")
    emb_layer = 20

    model_name = "500M_multi_species_v2"
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(emb_layer,),
        max_positions=1671,  # 1671
    )

    forward_fn = hk.transform(forward_fn)

    print("Tokenizing sequences")
    nptokens_fusion1 = tokenize_sequence_parallel(extr_key(fusion_data, "sequence1"), tokenizer.tokenize, 16)
    nptokens_fusion2 = tokenize_sequence_parallel(extr_key(fusion_data, "sequence2"), tokenizer.tokenize, 16)

    tokens_fusions1 = jnp.asarray([npt[1] for npt in nptokens_fusion1], dtype=jnp.int32)
    tokens_fusions2 = jnp.asarray([npt[1] for npt in nptokens_fusion2], dtype=jnp.int32)

    random_key = jax.random.PRNGKey(0)

    emb_pos = [tokens_fusions1.shape[1] // 2]
    print("Extracting embeddings for test sequences")

    total_start = time.time()

    emb_data1, metrics1 = embeding_nt(
        tokens_fusions1,
        4,
        emb_pos,
        forward_fn,
        f"embeddings_{emb_layer}",
        parameters,
        random_key,
        EMBD_TYPE,
    )
    emb_data2, metrics2 = embeding_nt(
        tokens_fusions2,
        4,
        emb_pos,
        forward_fn,
        f"embeddings_{emb_layer}",
        parameters,
        random_key,
        EMBD_TYPE,
    )

    total_time = time.time() - total_start

    print(f"Creating output folder at {OUTPUT_FOLDER}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CSV files")
    pd.DataFrame(emb_data1[:, 0, :]).to_csv(Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq1.csv", index=False, header=False)
    pd.DataFrame(emb_data2[:, 0, :]).to_csv(Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq2.csv", index=False, header=False)

    # Compute and save metrics using shared library
    metrics_summary = compute_metrics_summary(
        "Nucleotide Transformer",
        EMBD_TYPE,
        metrics1,
        metrics2,
        total_time,
        device_info=xla_bridge.get_backend().platform,
    )
    save_metrics(metrics_summary, OUTPUT_FOLDER, OUTPUT_NAME)

    print("\nProcessing completed successfully")


if __name__ == "__main__":
    main()
