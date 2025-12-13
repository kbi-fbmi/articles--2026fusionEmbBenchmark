import argparse
import os
import pickle as pkl

import torch

import ai


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    train = ai.load_fusion_actg(args.train_path, args.train_target)
    test_full = ai.load_fusion_actg(args.test_path, args.test_target)
    test, valid = ai.split_test(test_full)

    torch.manual_seed(args.seed)
    total_train_samples = train[0].shape[0]
    num_of_samples = args.num_of_samples if args.num_of_samples is not None else total_train_samples
    indices = torch.randperm(total_train_samples)
    x_train = (train[0][indices][:num_of_samples], train[1][indices][:num_of_samples])

    model = ai.get_fusionai_model()
    model, history = ai.train_adam(x_train, valid, model, num_epochs=args.epochs, verbose=True)

    model.save(os.path.join(args.output_folder, f"fusionai_model_{num_of_samples}.keras"))
    with open(os.path.join(args.output_folder, f"fusionai_history_{num_of_samples}.pkl"), "wb") as f:
        pkl.dump(history, f)
    evaluation_results = ai.evaluate_model(model, test)
    with open(os.path.join(args.output_folder, f"fusionai_results_{num_of_samples}.pkl"), "wb") as f:
        pkl.dump(evaluation_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--train-path", required=True, help="Path to training file")
    parser.add_argument("--train-target", required=True, help="Path to training target file")
    parser.add_argument("--test-path", required=True, help="Path to test  file")
    parser.add_argument("--test-target", required=True, help="Path to test target file")
    parser.add_argument("--output-folder", required=True, help="Folder to save outputs")
    parser.add_argument("--num-of-samples", type=int, default=None, help="Number of samples to use from training data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
