import argparse
from pathlib import Path

import pandas as pd
import torch
from evo2 import Evo2

from fbme_biolib import io
from fbme_biolib.fm import extr_key, tokenize_sequence_parallel


def mock_evo_call(tokens, return_embeddings=False, layer_names=None):
    print(f"Mock Evo2 call with tokens: {tokens.size()} and layer_names: {layer_names}")
    return [[], {"blocks.28.mlp.l3": torch.randn(tokens.size(0), tokens.size(1), 1024)}]


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
    # Configure logging

    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Process Evo2 embeddings.")
    # parser.add_argument("--path_train", required=True, help="Path to the training data file")
    # parser.add_argument("--path_test", required=True, help="Path to the testing data file")
    # parser.add_argument("--output_folder", required=True, help="Output folder for saving results")
    # args = parser.parse_args()

    # PATH_TRAIN = args.path_train
    # PATH_TEST = args.path_test
    # OUTPUT_FOLDER = args.output_folder

    PATH_TRAIN = "/mnt/e/Data/Fuse/fusionai_train_sim_107.txt"
    PATH_TEST = "/mnt/e/Data/Fuse/fusionai_test_sim_107.txt"
    OUTPUT_FOLDER = "./ouput"

    print(f"Loading training data from {PATH_TRAIN}")
    fusion_train = io.load_fusions_from_fusionaitxt(PATH_TRAIN)
    print(f"Loading testing data from {PATH_TEST}")
    fusion_test = io.load_fusions_from_fusionaitxt(PATH_TEST)

    print("Initializing Evo2 model")
    evo2_model = Evo2("evo2_7b")

    print("Tokenizing sequences")
    nptokens_fusion_test1 = tokenize_sequence_parallel(
        extr_key(fusion_test, "sequence1"), evo2_model.tokenizer.tokenize, 32
    )
    nptokens_fusion_test2 = tokenize_sequence_parallel(
        extr_key(fusion_test, "sequence2"), evo2_model.tokenizer.tokenize, 32
    )
    nptokens_fusion_train1 = tokenize_sequence_parallel(
        extr_key(fusion_train, "sequence1"), evo2_model.tokenizer.tokenize, 32
    )
    nptokens_fusion_train2 = tokenize_sequence_parallel(
        extr_key(fusion_train, "sequence2"), evo2_model.tokenizer.tokenize, 32
    )

    tokens_fusion_test1 = torch.tensor(nptokens_fusion_test1, dtype=torch.int).to(
        "cuda:0"
    )
    tokens_fusion_test2 = torch.tensor(nptokens_fusion_test2, dtype=torch.int).to(
        "cuda:0"
    )
    tokens_fusion_train1 = torch.tensor(nptokens_fusion_train1, dtype=torch.int).to(
        "cuda:0"
    )
    tokens_fusion_train2 = torch.tensor(nptokens_fusion_train2, dtype=torch.int).to(
        "cuda:0"
    )

    emb_pos = [0, tokens_fusion_test1.size(1) // 2, tokens_fusion_test1.size(1) - 1]
    print("Extracting embeddings for test sequences")
    emb_test1 = embeding_evo(
        tokens_fusion_test1, 64, emb_pos, mock_evo_call, layer_name="blocks.28.mlp.l3"
    )
    emb_test2 = embeding_evo(
        tokens_fusion_test2, 64, emb_pos, mock_evo_call, layer_name="blocks.28.mlp.l3"
    )

    emb_pos = [0, tokens_fusion_train1.size(1) // 2, tokens_fusion_test1.size(1) - 1]
    print("Extracting embeddings for train sequences")
    emb_train1 = embeding_evo(
        tokens_fusion_train1, 64, emb_pos, mock_evo_call, layer_name="blocks.28.mlp.l3"
    )
    emb_train2 = embeding_evo(
        tokens_fusion_train2, 64, emb_pos, mock_evo_call, layer_name="blocks.28.mlp.l3"
    )

    print(f"Creating output folder at {OUTPUT_FOLDER}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CSV files")
    pd.DataFrame(emb_test1[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_test_seq1_left.csv", index=False, header=False
    )
    pd.DataFrame(emb_test1[:, 1, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_test_seq1_middle.csv", index=False, header=False
    )
    pd.DataFrame(emb_test1[:, 2, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_test_seq1_right.csv", index=False, header=False
    )

    pd.DataFrame(emb_test2[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_test_seq2_left.csv", index=False, header=False
    )
    pd.DataFrame(emb_test2[:, 1, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_test_seq2_middle.csv", index=False, header=False
    )
    pd.DataFrame(emb_test2[:, 2, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_test_seq2_right.csv", index=False, header=False
    )

    pd.DataFrame(emb_train1[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_train_seq1_left.csv", index=False, header=False
    )
    pd.DataFrame(emb_train1[:, 1, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_train_seq1_middle.csv", index=False, header=False
    )
    pd.DataFrame(emb_train1[:, 2, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_train_seq1_right.csv", index=False, header=False
    )

    pd.DataFrame(emb_train2[:, 0, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_train_seq2_left.csv", index=False, header=False
    )
    pd.DataFrame(emb_train2[:, 1, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_train_seq2_middle.csv", index=False, header=False
    )
    pd.DataFrame(emb_train2[:, 2, :].numpy()).to_csv(
        Path(OUTPUT_FOLDER) / "evo_train_seq2_right.csv", index=False, header=False
    )

    print("Processing completed successfully")


if __name__ == "__main__":
    main()
