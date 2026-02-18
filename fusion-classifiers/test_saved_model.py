import argparse
import os
import pickle as pkl
import sys
from pathlib import Path

import fmlib.io as fmio
import torch
from fmlib.embd import load_fusion_embeddings

import ai as ai


def load_test_data(seq1_path, seq2_path, target_path):
    """
    Load test data from custom seq1, seq2, and target files.
    
    Args:
        seq1_path (str or Path): Path to sequence 1 embeddings CSV file
        seq2_path (str or Path): Path to sequence 2 embeddings CSV file
        target_path (str or Path): Path to target labels CSV file
    
    Returns:
        tuple: (x_test, y_test) as torch tensors
    """
    test_data = load_fusion_embeddings(
        Path(seq1_path),
        Path(seq2_path),
        Path(target_path),
    )
    print(f"Test data shape: {test_data[0].shape}")
    print(f"Test labels shape: {test_data[1].shape}")
    return test_data


def load_saved_model(model_path, model_type="keras"):
    """
    Load a saved model from disk.
    
    Args:
        model_path (str or Path): Path to the saved model file
        model_type (str): Type of model ('keras' for .keras files, 'svm' for .pkl files)
    
    Returns:
        The loaded model
    """
    if model_type == "keras":
        import keras as kr
        model = kr.models.load_model(model_path)
        print(f"Loaded Keras model from: {model_path}")
    elif model_type == "svm":
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        print(f"Loaded SVM model from: {model_path}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def test_keras_model(model, test_data):
    """
    Test a Keras model on the provided test data.
    
    Args:
        model: Keras model
        test_data (tuple): (x_test, y_test) as torch tensors
    
    Returns:
        dict: Evaluation results
    """
    print("\nEvaluating Keras model...")
    evaluation_results = ai.evaluate_model(model, test_data)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {evaluation_results['Accuracy']:.2f}%")
    print(f"Precision: {evaluation_results['Precision']:.4f}")
    print(f"Recall:    {evaluation_results['Recall']:.4f}")
    print(f"F1 Score:  {evaluation_results['F1 Score']:.4f}")
    print(f"ROC AUC:   {evaluation_results['ROC AUC']:.4f}")
    print("\nConfusion Matrix:")
    print(evaluation_results['Confusion Matrix'])
    print("="*60)
    
    return evaluation_results


def test_svm_model(model_results, test_data):
    """
    Test an SVM model on the provided test data.
    
    Args:
        model_results: Dictionary containing SVM model and results
        test_data (tuple): (x_test, y_test) as torch tensors
    
    Returns:
        dict: Evaluation results
    """
    print("\nEvaluating SVM model...")
    
    x_test, y_test = test_data
    x_test = x_test.numpy()
    y_test = torch.argmax(y_test, dim=1).numpy()
    
    # Extract the trained model from the results dictionary
    if 'models' in model_results:
        svm_model = model_results['models'][0]  # Get first kernel's model
    elif 'model' in model_results:
        svm_model = model_results['model']
    else:
        raise ValueError("Could not find SVM model in loaded file")
    
    # Make predictions
    y_pred = svm_model.predict(x_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("="*60)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }


def save_results(output_file, evaluation_results):
    """
    Save evaluation results to a pickle file.
    
    Args:
        output_file (str or Path): Path to save the results
        evaluation_results (dict): Evaluation results to save
    """
    with open(output_file, "wb") as f:
        pkl.dump(evaluation_results, f)
    print(f"\nResults saved to: {output_file}")


def main(args):
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.seq1, args.seq2, args.target)
    
    # Determine model type from file extension
    model_path = Path(args.model_path)
    if model_path.suffix == '.keras':
        model_type = 'keras'
        model = load_saved_model(model_path, model_type)
        evaluation_results = test_keras_model(model, test_data)
    elif model_path.suffix == '.pkl':
        model_type = 'svm'
        model = load_saved_model(model_path, model_type)
        evaluation_results = test_svm_model(model, test_data)
    else:
        raise ValueError(f"Unsupported model file type: {model_path.suffix}. Use .keras or .pkl")
    
    # Save results if output file is specified
    if args.output:
        save_results(args.output, evaluation_results)
    
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a saved model on custom seq1, seq2, and target files"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved model file (.keras for neural network, .pkl for SVM)",
    )
    parser.add_argument(
        "--seq1",
        required=True,
        help="Path to sequence 1 embeddings CSV file",
    )
    parser.add_argument(
        "--seq2",
        required=True,
        help="Path to sequence 2 embeddings CSV file",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target labels CSV file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save evaluation results (pickle file). If not specified, results won't be saved.",
    )
    
    args = parser.parse_args()
    
    # If no arguments are provided (e.g. running from IDE), set defaults here
    if len(sys.argv) == 1:
        args = parser.parse_args(
            [
                "--model-path",
                "models_output/nt_mean_model_36302.keras",
                "--seq1",
                "../notebooks/download/embeddings/nt_test_seq1.csv",
                "--seq2",
                "../notebooks/download/embeddings/nt_test_seq2.csv",
                "--target",
                "../notebooks/download/embeddings/fusionai_test_target.csv",
                # "--output", "test_results.pkl",  # Uncomment to save results
            ]
        )
    
    main(args)
