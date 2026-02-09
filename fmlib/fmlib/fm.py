"""Faundantion model helpers functions"""

import concurrent.futures
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def hello():
    print("Hello, this is a helper function!")


def compute_metrics_summary(
    model_name: str,
    embd_type: str,
    metrics1: dict,
    metrics2: dict,
    total_time: float,
    device_info: str = None,
) -> dict:
    """
    Compute performance metrics summary for embedding extraction.
    
    Parameters
    ----------
    model_name : str
        Name of the model used for embedding extraction.
    embd_type : str
        Type of embedding extraction (e.g., 'middle', 'mean').
    metrics1 : dict
        Metrics from first sequence processing.
    metrics2 : dict
        Metrics from second sequence processing.
    total_time : float
        Total processing time in seconds.
    device_info : str, optional
        Device information (GPU name, TPU, etc.).
    
    Returns
    -------
    dict
        Dictionary containing comprehensive performance metrics.
    """
    all_batch_times = metrics1["batch_times"] + metrics2["batch_times"]
    total_samples = metrics1["total_samples"] + metrics2["total_samples"]
    
    metrics_summary = {
        "model": model_name,
        "embd_type": embd_type,
        "total_samples": total_samples,
        "total_time_seconds": total_time,
        "inference_latency_ms_per_sample": (sum(all_batch_times) / total_samples) * 1000,
        "throughput_samples_per_second": total_samples / total_time,
        "avg_batch_time_seconds": sum(all_batch_times) / len(all_batch_times),
        "min_batch_time_seconds": min(all_batch_times),
        "max_batch_time_seconds": max(all_batch_times),
    }
    
    # Add VRAM metrics if available
    if "peak_vram_mb" in metrics1 or "peak_vram_mb" in metrics2:
        metrics_summary["peak_vram_mb"] = max(
            metrics1.get("peak_vram_mb", 0),
            metrics2.get("peak_vram_mb", 0)
        )
    
    # Add device info
    if device_info:
        metrics_summary["device"] = device_info
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        metrics_summary["device"] = torch.cuda.get_device_name()
    
    return metrics_summary


def save_metrics(metrics_summary: dict, output_folder: str, output_name: str, verbose: bool = True):
    """
    Save performance metrics to JSON file and optionally print summary.
    
    Parameters
    ----------
    metrics_summary : dict
        Dictionary containing performance metrics.
    output_folder : str
        Path to output folder.
    output_name : str
        Base name for output files.
    verbose : bool, optional
        Whether to print metrics summary (default is True).
    """
    metrics_path = Path(output_folder) / f"{output_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    if verbose:
        print("\n=== Performance Metrics ===")
        print(f"Model: {metrics_summary['model']}")
        print(f"Total Samples: {metrics_summary['total_samples']}")
        print(f"Inference Latency: {metrics_summary['inference_latency_ms_per_sample']:.2f} ms/sample")
        print(f"Throughput: {metrics_summary['throughput_samples_per_second']:.2f} samples/sec")
        if "peak_vram_mb" in metrics_summary:
            print(f"Peak VRAM: {metrics_summary['peak_vram_mb']:.2f} MB")
        if "device" in metrics_summary:
            print(f"Device: {metrics_summary['device']}")
        print(f"Metrics saved to: {metrics_path}")


def tokenize_sequence_parallel(seq, tokenize_function, max_workers=16) -> list:
    """
    Tokenize a sequence in parallel using the provided tokenize_function.

    Parameters
    ----------
    seq : iterable
        The sequence to tokenize.
    tokenize_function : callable
        The function to apply to each element of the sequence.
    max_workers : int, optional
        The maximum number of worker threads to use (default is 16).

    Returns
    -------
    list
        A list of tokenized elements.
    """
    print(f"Starting tokenization with {max_workers} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        tokens = list(executor.map(tokenize_function, seq))
    print("Tokenization completed")
    return tokens


def extr_key(dict_list, key):
    """
    Extract values for a specific key from a list of dictionaries.
    
    Parameters
    ----------
    dict_list : list of dict
        List of dictionaries to extract from.
    key : str
        Key to extract from each dictionary.
    
    Returns
    -------
    list
        List of values for the specified key.
    """
    print(f"Extracting key '{key}' from dictionary list")
    return [d[key] for d in dict_list]
