# Fusion Classifiers

Classification models for gene fusion detection using embeddings from genomic foundation models. This module provides training scripts and model architectures for both embedding-based classifiers and the FusionAI reimplementation.

## Overview

This project implements multiple approaches to classify gene fusions:

1. **Embedding-based classifiers**: Neural networks trained on precomputed embeddings from foundation models (DNABERT, NT, Hyena, EVO2)
2. **FusionAI reimplementation**: Original FusionAI CNN architecture using one-hot encoded DNA sequences

Both approaches use the same training/testing data from FusionGDB2 for fair comparison.

## Project Structure

```
fusion-classifiers/
├── train_embeddings.py       # Train models on foundation model embeddings
├── train_fusionai.py         # Train original FusionAI architecture
├── ai.py                     # Core model architectures and utilities
├── train_script_example.sh   # Example training script
└── README.md
```

## Models

### 1. Embedding-based Classifier

A feedforward neural network that takes concatenated embeddings from two fusion sequences:

**Architecture:**
- Input: Concatenated embeddings from seq1 and seq2
- Dense layer (256 units, ReLU)
- Dropout (0.4)
- Dense layer (64 units, ReLU)
- Dropout (0.4)
- Output layer (softmax for classification)

**Input format:** Precomputed embeddings (e.g., from DNABERT, NT, Hyena, EVO2)

### 2. FusionAI Model

Convolutional Neural Network for one-hot encoded DNA sequences:

**Architecture:**
- Input: One-hot encoded DNA (shape: length × 4 × 1)
- Conv2D (256 filters, 20×4 kernel, ReLU)
- Conv2D (32 filters, 200×1 kernel, ReLU)
- MaxPooling2D (20×1 pool size)
- Dropout (0.25)
- Flatten
- Dense (32 units, ReLU)
- Dropout (0.4)
- Output layer (softmax)

**Input format:** One-hot encoded DNA sequences (ACGT)

## Quick Start

### Prerequisites

- Python 3.11+
- GPU recommended for faster training

### Setup

1. **Initialize environment**
   ```bash
   cd fusion-classifiers
   uv sync --python 3.11
   ```

2. **Activate environment**
   ```bash
   source .venv/bin/activate
   ```

## Usage

### Training Embedding-based Classifiers

Train a model using precomputed embeddings:

```bash
python train_embeddings.py \
    --epochs 200 \
    --embedding-type nt \
    --data-folder ../notebooks/download/embeddings \
    --output-folder models_output \
    --num-of-samples 1000 \
    --seed 42
```

**Parameters:**
- `--epochs`: Number of training epochs
- `--embedding-type`: Foundation model type (`nt`, `evo`, `hyena`, `bert`)
- `--data-folder`: Path to embeddings folder
- `--output-folder`: Where to save trained models
- `--num-of-samples`: Number of training samples (optional, uses all if not specified)
- `--seed`: Random seed for reproducibility

### Training FusionAI Model

Train the FusionAI CNN on one-hot encoded sequences:

```bash
python train_fusionai.py \
    --epochs 200 \
    --train-path data/fg_newdata_train.txt \
    --train-target data/train_target.csv \
    --test-path data/fg_newdata_test.txt \
    --test-target data/test_target.csv \
    --output-folder models_output \
    --num-of-samples 1000 \
    --seed 42
```

**Parameters:**
- `--epochs`: Number of training epochs
- `--train-path`: Path to training sequences file
- `--train-target`: Path to training labels file
- `--test-path`: Path to test sequences file
- `--test-target`: Path to test labels file
- `--output-folder`: Where to save trained models
- `--num-of-samples`: Number of training samples (optional)
- `--seed`: Random seed for reproducibility

### Using the Example Script

An example training script is provided:

```bash
# Edit train_script_example.sh to configure paths and parameters
bash train_script_example.sh
```

## Output Files

Training produces three output files per model:

1. **Model file**: `<prefix>_model_<samples>.keras` - Trained Keras model
2. **History file**: `<prefix>_history_<samples>.pkl` - Training history (loss, accuracy curves)
3. **Results file**: `<prefix>_results_<samples>.pkl` - Evaluation metrics on test set

### Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-score**: Weighted average F1
- **AUC-ROC**: Area under ROC curve (when applicable)
- **Confusion Matrix**: Detailed classification breakdown

## API Reference

### Core Functions (ai.py)

#### Model Creation

```python
# Get embedding-based classifier
model = ai.get_simple_model(input_dim=2048, num_classes=2)

# Get FusionAI CNN model
model = ai.get_fusionai_model(input_dim=20000, num_classes=2)
```

#### Training

```python
# Train with Adam optimizer and early stopping
model, history = ai.train_adam(
    data_train=(x_train, y_train),
    data_val=(x_val, y_val),
    model=model,
    num_epochs=200,
    batch_size=256,
    verbose=True
)
```

#### Evaluation

```python
# Evaluate on test set
results = ai.evaluate_model(model, (x_test, y_test))
# Returns: accuracy, precision, recall, f1, auc, confusion matrix
```

#### Data Loading

```python
# Load embeddings
train, valid, test = ai.load_and_prepare_data(data_folder, prefix='nt')

# Load one-hot encoded sequences
train = ai.load_fusion_actg(sequences_path, target_path)

# Split test into validation and final test
test, valid = ai.split_test(test_full)
```

## Data Format

### Embedding Files

CSV files containing precomputed embeddings:
- `<prefix>_train_seq1.csv`: Training sequence 1 embeddings
- `<prefix>_train_seq2.csv`: Training sequence 2 embeddings
- `fusionai_train_target.csv`: Training labels

Where `<prefix>` is one of: `nt`, `evo`, `hyena`, `bert`

### FusionAI Input Files

Tab-separated text files with columns:
- Gene names, chromosomes, positions, strands
- DNA sequence 1
- DNA sequence 2
- Target label

## Dependencies

Key packages (see [pyproject.toml](pyproject.toml) for full list):
- **keras**: Model architecture and training (3.10+)
- **tensorflow/tf-keras**: Backend for Keras (2.19+)
- **torch**: PyTorch for tensor operations (2.7+)
- **scikit-learn**: Evaluation metrics and utilities (1.7+)
- **pandas**: Data manipulation
- **fmlib**: Core fusion gene utilities (local package)

## Training Tips

### Performance Optimization

1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES` for GPU training
2. **Batch size**: Adjust based on available memory (default: 256)
3. **Early stopping**: Built-in with patience=10 epochs

### Sample Efficiency Testing

Train with different sample sizes to evaluate data efficiency:

```bash
for samples in 100 500 1000 5000 10000; do
    python train_embeddings.py \
        --epochs 200 \
        --embedding-type nt \
        --data-folder ../notebooks/download/embeddings \
        --output-folder models_output \
        --num-of-samples $samples
done
```

### Reproducibility

Always set `--seed` parameter for reproducible results:
```bash
--seed 42
```

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: Edit `batch_size` parameter in training functions
- Use fewer samples: Set `--num-of-samples` to a smaller value
- Train on CPU: Unset `CUDA_VISIBLE_DEVICES`

### Keras Backend Issues
The code uses PyTorch as Keras backend. This is set in [ai.py](ai.py):
```python
os.environ["KERAS_BACKEND"] = "torch"
```

### Missing Embeddings
Download precomputed embeddings from [Zenodo](https://zenodo.org/records/17898581) or generate them using the fmextractors modules.

## Related Documentation

- [Main Project README](../README.md)
- [fmlib Documentation](../fmlib/README.md)
- [Notebooks Analysis](../notebooks/README.md)
- [Embedding Extractors](../fmextractors/)

## Citation

If you use this code, please cite:
- FusionGDB2: https://compbio.uth.edu/FusionGDB2/
- Original FusionAI paper
- Respective foundation model papers (DNABERT, NT, Hyena, EVO2)

## License

See LICENSE file for details.
