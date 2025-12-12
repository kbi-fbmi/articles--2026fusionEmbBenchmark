import argparse
import os
import pickle as pkl
import sys
from pathlib import Path

import pandas as pd
import torch

import ai


def load_and_prepare_data(data_folder, prefix):
    train = ai.load_fusion_embedings(
        data_folder / f"{prefix}_train_seq1.csv",
        data_folder / f"{prefix}_train_seq2.csv",
        data_folder / "fusionai_train_target.csv",
    )
    test_full = ai.load_fusion_embedings(
        data_folder / f"{prefix}_test_seq1.csv",
        data_folder / f"{prefix}_test_seq2.csv",
        data_folder / "fusionai_test_target.csv",
    )
    test, valid = ai.split_test(test_full)
    print(f"Training data shape: {train[0].shape}")
    print(f"Validation data shape: {valid[0].shape}")
    print(f"Test data shape: {test[0].shape}")
    return train, valid, test


def sample_training_data(train, num_of_samples, seed=42):
    """
    Randomly samples a subset of training data from the provided dataset.

    Args:
        train (tuple of torch.Tensor): A tuple containing training data and labels,
            where train[0] is the data tensor and train[1] is the labels tensor.
        num_of_samples (int or None): The number of samples to select from the training data.
            If None, all samples are used.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple of torch.Tensor: A tuple (x_train, y_train) containing the sampled data and labels.

    Notes:
        - If num_of_samples is less than the total number of training samples, a random subset is selected.
        - If num_of_samples is greater than or equal to the total number of training samples, the entire dataset is returned.
    """

    torch.manual_seed(seed)
    total_train_samples = train[0].shape[0]
    num_of_samples = num_of_samples if num_of_samples is not None else total_train_samples

    x_train = train

    if num_of_samples < total_train_samples:
        print(f"Sampling {num_of_samples} from {total_train_samples} training samples")
        indices = torch.randperm(total_train_samples)
        x_train = (train[0][indices][:num_of_samples], train[1][indices][:num_of_samples])

    return x_train


def save_results(output_folder, embedding_type, num_of_samples, model, history, evaluation_results):
    model_filename = f"{embedding_type}_model_{num_of_samples}.keras"
    history_filename = f"{embedding_type}_history_{num_of_samples}.pkl"
    results_filename = f"{embedding_type}_results_{num_of_samples}.pkl"

    model.save(os.path.join(output_folder, model_filename))
    with open(os.path.join(output_folder, history_filename), "wb") as f:
        pkl.dump(history, f)
    with open(os.path.join(output_folder, results_filename), "wb") as f:
        pkl.dump(evaluation_results, f)


def main(args):
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    prefix = args.embedding_type
    num_of_samples = args.num_of_samples

    print(f"Loading embeddings from: {prefix}")
    data_folder = Path(args.data_folder)

    train, valid, test = load_and_prepare_data(data_folder, prefix)
    x_train = sample_training_data(train, num_of_samples)
    num_of_samples = x_train[0].shape[0]

    input_dim = x_train[0].shape[1]
    print(f"Input dimension: {input_dim}")
    model = ai.get_fully_connected(input_dim)

    print(f"Starting training for {args.epochs} epochs...")
    model, history = ai.train_adam(x_train, valid, model, num_epochs=args.epochs, verbose=True)

    print("Evaluating model...")
    evaluation_results = ai.evaluate_model(model, test)

    save_results(args.output_folder, prefix, num_of_samples, model, history, evaluation_results)

    # SVM linear
    custom_kernels = [{"name": "Linear", "kernel": "linear", "C": 1.0}]
    evaluation_results_svm = {}

    print("\n Evaluating Model with SVM:")
    evaluation_results_svm = ai.train_and_evaluate_svm(x_train, test, valid, max_samples=40000, kernels=custom_kernels)
    with open(os.path.join(args.output_folder, f"{prefix}_svm_linear_{num_of_samples}.pkl"), "wb") as f:
        pkl.dump(evaluation_results_svm, f)

    # SVM RBF
    custom_kernels = [{"name": "RBF", "kernel": "rbf", "C": 1.0, "gamma": "scale"}]
    evaluation_results_svm = ai.train_and_evaluate_svm(x_train, test, valid, max_samples=40000, kernels=custom_kernels)
    with open(os.path.join(args.output_folder, f"{prefix}_svm_rbf_{num_of_samples}.pkl"), "wb") as f:
        pkl.dump(evaluation_results_svm, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network on specific embeddings")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument(
        "--embedding-type",
        required=True,
        choices=["nt", "evo", "bert", "hyena"],
        help="Type of embedding (data prefix) to use (nt, evo, bert, hyena)",
    )
    parser.add_argument("--data-folder", required=True, help="Path to data folder containing embeddings")
    parser.add_argument("--output-folder", required=True, help="Folder to save outputs")
    parser.add_argument(
        "--num-of-samples",
        type=int,
        default=None,
        help="Number of samples to use from training data (default: use all)",
    )
    args = parser.parse_args()

    # If no arguments are provided (e.g. running from IDE), set defaults here
    if len(sys.argv) == 1:
        args = parser.parse_args(
            [
                "--epochs",
                "300",
                "--embedding-type",
                "nt",
                "--data-folder",
                "/home/user/fmfusions/data",
                "--output-folder",
                "/home/user/fmfusions/output",
                # "--num-of-samples", "1000",  # Uncomment if you want to set this
            ]
        )

    main(args)
