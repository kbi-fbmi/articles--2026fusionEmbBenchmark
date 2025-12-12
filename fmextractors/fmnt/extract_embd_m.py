import argparse
import concurrent.futures
from pathlib import Path
from typing import Any, Callable, List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from fmlib import io
from fmlib.fm import extr_key, tokenize_sequence_parallel
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
) -> np.ndarray:
    print(f"Starting embedding extraction for layer: {layer_name}")

    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)
    embeddings: np.ndarray | None = None
    for i in range(num_batches):
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size]
        print(f"Processing batch {i + 1}/{num_batches} with {len(batch_tokens)} tokens")
        batch_embeddings = nt_model.apply(parameters, random_key, batch_tokens)
        selected_embeddings = batch_embeddings[layer_name][:, emb_positions, :]
        embeddings = (
            np.concatenate((embeddings, selected_embeddings), axis=0) if embeddings is not None else selected_embeddings
        )
        print(f"Batch {i + 1} done")

    print("Embedding extraction completed")
    return embeddings


def main():
    # Configure logging

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process NT embeddings.")
    parser.add_argument("--path_data", required=True, help="Path to the  data file")
    parser.add_argument("--output_folder", required=True, help="Output folder for saving results")
    parser.add_argument("--output_name", required=True, help="Output prefix for saving results")
    args = parser.parse_args()

    PATH_DATA = args.path_data
    OUTPUT_FOLDER = args.output_folder
    OUTPUT_NAME = args.output_name

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
    emb_data1 = embeding_nt(tokens_fusions1, 4, emb_pos, forward_fn, f"embeddings_{emb_layer}", parameters, random_key)
    emb_data2 = embeding_nt(tokens_fusions2, 4, emb_pos, forward_fn, f"embeddings_{emb_layer}", parameters, random_key)

    print(f"Creating output folder at {OUTPUT_FOLDER}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CSV files")
    pd.DataFrame(emb_data1[:, 0, :]).to_csv(Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq1.csv", index=False, header=False)
    pd.DataFrame(emb_data2[:, 0, :]).to_csv(Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq2.csv", index=False, header=False)

    print("Processing completed successfully")


if __name__ == "__main__":
    main()
