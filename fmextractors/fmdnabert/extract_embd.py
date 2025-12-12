# import argparse
import argparse
import concurrent.futures
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fmlib import io
from fmlib.fm import extr_key
from transformers import AutoModel, AutoTokenizer

emb_counter = 0

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)


def tokenize_sequence_parallel(seq, tokenize_function, max_workers=16):
    print(f"Starting tokenization with {max_workers} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        results = executor.map(tokenize_function, seq)
        tokens, emb_positions = zip(*results)

    print("Tokenization completed")
    return list(tokens), list(emb_positions)


def get_embeddings(tokens, model):
    # Calculate the embeddings
    print(f"embedding token size:{tokens.size(1)}")
    hidden_states = model(tokens)[0].detach().cpu().numpy()  # [1, sequence_length, 768]

    return hidden_states


def embeding_dna_bert(tokens, batch_size, emb_positions, model):
    print(f"Starting embedding extraction")

    num_batches = len(tokens) // batch_size + (len(tokens) % batch_size > 0)
    embeddings = None
    for i in range(num_batches):
        batch_tokens = tokens[i * batch_size : (i + 1) * batch_size]
        # batch_tokens = batch_tokens.to("cuda:0")
        print(f"Processing batch {i + 1}/{num_batches} with {len(batch_tokens)} tokens")
        batch_embeddings = get_embeddings(batch_tokens, model)

        selected_embeddings = np.array(
            [batch_embeddings[j, emb_positions[i * batch_size + j], :] for j in range(len(batch_tokens))]
        )
        embeddings = (
            np.concatenate((embeddings, selected_embeddings), axis=0) if embeddings is not None else selected_embeddings
        )
        print(f"Batch {i + 1} done")

    print("Embedding extraction completed")
    return embeddings


def padding_tokens(token, padding_size=2143, front_value=1, backvalue=2):
    # Pad the token tensor to the specified size
    if padding_size < token.size(1):
        raise ValueError(f"Padding size {padding_size} is less than the token size {token.size(1)}")
    tsz = token.size(1)
    pad_front = math.floor((padding_size - token.size(1)) / 2)
    pad_back = math.ceil((padding_size - token.size(1)) / 2)
    token = torch.cat(
        [
            torch.full((1, pad_front), front_value),
            token,
            torch.full((1, pad_back), backvalue),
        ],
        dim=1,
    )
    back = token.size(1) - pad_back - 2
    half = token.size(1) // 2

    return (token, np.asarray([pad_front + 1, half, back]))


def tokenize_dna(dna):
    return padding_tokens(tokenizer(dna, return_tensors="pt")["input_ids"])


def main():
    parser = argparse.ArgumentParser(description="Process DNABERT embeddings.")
    parser.add_argument("--path_data", required=True, help="Path to the  data file")
    parser.add_argument("--output_folder", required=True, help="Output folder for saving results")
    parser.add_argument("--output_name", required=True, help="Output prefix for saving results")
    args = parser.parse_args()

    PATH_DATA = args.path_data
    OUTPUT_FOLDER = args.output_folder
    OUTPUT_NAME = args.output_name

    # PATH_DATA = "PATH_TO_YOUR_DATA.txt"
    # OUTPUT_FOLDER = "./ouput"
    # OUTPUT_NAME = "bert_train"

    print(f"Loading training data from {PATH_DATA}")
    fusion_data = io.load_fusions_from_fusionaitxt(PATH_DATA)

    print("Loading DNABERT-2 model from Hugging Face")
    dnabert2_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to("cuda:0")

    print("Tokenizing sequences")
    nptokens_fusion1, emb_positions = tokenize_sequence_parallel(extr_key(fusion_data, "sequence1"), tokenize_dna, 32)
    nptokens_fusion1 = torch.concatenate(nptokens_fusion1).to("cuda:0")
    emb_positions1 = np.asarray(emb_positions)

    nptokens_fusion2, emb_positions = tokenize_sequence_parallel(extr_key(fusion_data, "sequence2"), tokenize_dna, 32)
    nptokens_fusion2 = torch.concatenate(nptokens_fusion2).to("cuda:0")
    emb_positions2 = np.asarray(emb_positions)

    emb_data1 = embeding_dna_bert(nptokens_fusion1, 2, emb_positions1, dnabert2_model)
    emb_data2 = embeding_dna_bert(nptokens_fusion2, 2, emb_positions2, dnabert2_model)
    print("Extracting embeddings for test sequences")

    print(f"Creating output folder at {OUTPUT_FOLDER}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CSV files")
    pd.DataFrame(emb_data1[:, 1, :]).to_csv(Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq1.csv", index=False, header=False)
    pd.DataFrame(emb_data2[:, 1, :]).to_csv(Path(OUTPUT_FOLDER) / f"{OUTPUT_NAME}_seq2.csv", index=False, header=False)

    print("Processing completed successfully")


if __name__ == "__main__":
    main()
