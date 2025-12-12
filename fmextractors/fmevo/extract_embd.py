import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fmlib import io
from fmlib.fm import extr_key, tokenize_sequence_parallel

from evo2 import Evo2


def evo_tokenize(sequence):
    return list(np.frombuffer(sequence.encode("utf-8"), dtype=np.uint8))


def mock_evo_call(tokens, return_embeddings=False, layer_names=None):
    print(f"Mock Evo2 call with tokens: {tokens.size()} and layer_names: {layer_names}")
    return {"blocks.28.mlp.l3": torch.randn(tokens.size(0), tokens.size(1), 1024)}


def embeding_evo(
    tokens, batch_size, emb_positions, evo2_model, layer_name="blocks.28.mlp.l3"
):
    print(f"Starting embedding extraction for layer: {layer_name}")
    embeddings = torch.asarray([])
    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)

    for i in range(num_batches):
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size]
        print(f"Processing batch {i + 1}/{num_batches} with {len(batch_tokens)} tokens")
        _, batch_embeddings = evo2_model(
            batch_tokens, return_embeddings=True, layer_names=[layer_name]
        )
        selected_embeddings = batch_embeddings[layer_name][:, emb_positions, :]
        embeddings = torch.cat((embeddings, selected_embeddings.cpu()), dim=0)
        print(f"Batch {i + 1} done")

    print("Embedding extraction completed")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Process Evo2 embeddings.")
    parser.add_argument("--path_data", required=True, help="Path to the  data file")
    parser.add_argument(
        "--output_folder", required=True, help="Output folder for saving results"
    )
    parser.add_argument(
        "--output_name", required=True, help="Output prefix for saving results"
    )
    args = parser.parse_args()

    PATH_DATA = args.path_data
    OUTPUT_FOLDER = args.output_folder
    OUTPUT_NAME = args.output_name

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
    emb1 = embeding_evo(
        tokens_fusion1, 4, emb_pos, evo2_model, layer_name="blocks.28.mlp.l3"
    )
    emb2 = embeding_evo(
        tokens_fusion2, 4, emb_pos, evo2_model, layer_name="blocks.28.mlp.l3"
    )

    print(f"Creating output folder at {OUTPUT_FOLDER}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CSV files")
    pd.DataFrame(emb1[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq1.csv", index=False, header=False
    )

    pd.DataFrame(emb2[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq2.csv", index=False, header=False
    )

    print("Processing completed successfully")


if __name__ == "__main__":
    main()
