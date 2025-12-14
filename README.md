# Benchmarking Genomic Foundation Models for Gene Fusion Detection

This repository contains benchmarking framework for evaluating multiple genomic foundation models on gene fusion detection from DNA sequences. The project implements embeddings extraction from various state-of-the-art foundation models and trains classification models to detect fusion genes.

## Overviewls

Gene fusions are hybrid genes formed from portions of two different genes and are important biomarkers in cancer and other diseases. This project evaluates and compares the performance of modern genomic foundation models in detecting gene fusions from DNA sequences.

## Results

Comprehensive benchmarking results comparing model performance on gene fusion detection are available in the notebooks. See the [analysis notebook](https://github.com/kbi-fbmi/articles--2026fusionEmbBenchmark/blob/main/notebooks/compare_models.ipynb) for detailed comparisons.

### Key Components

- **fmlib**: Core biomedical library with utilities for data loading, preprocessing, and sequence handling
- **fmextractors**: Embedding extraction modules for multiple foundation models
- **fusion-classifiers**: Classification models using embeddings as features
- **notebooks**: Analysis and visualization notebooks

## Models Benchmarked

The following genomic foundation models are benchmarked:

| Model | Description | Status |
|-------|-------------|--------|
| **DNABERT** | Domain-specific BERT model trained on DNA sequences | ✓ |
| **Nucleotide Transformer (NT)** | Large-scale protein language model adapted for DNA | ✓ |
| **Hyena** | Efficient long-range sequence modeling | ✓ |
| **EVO2** | Evolution-based genomic foundation model | ✓ |

## Data
- **Dataset from FusionAI project**:[FusionAI data homepage](https://compbio.uth.edu/FusionGDB2/FusionAI/)
- **Training Data**: [FusionGDB2 training set](https://compbio.uth.edu/FusionGDB2/FusionAI/fg_newdata_train.txt)
- **Testing Data**: [FusionGDB2 testing set](https://compbio.uth.edu/FusionGDB2/FusionAI/fg_newdata_test.txt)
- **Precomputed Embeddings**: Available at [Zenodo](https://zenodo.org/records/17898581)

Embeddings are computed from DNA sequences and serve as input features for downstream classification models.

## Project Structure

```
fmfusions/
├── fmlib/                      # Core library
│   ├── fm.py                   # Foundation model utilities
│   ├── io.py                   # Data I/O functions
│   └── tests/                  # tests
├── fmextractors/               # Embedding extraction modules
│   ├── fmdnabert/              # DNABERT embeddings
│   ├── fmnt/                   # Nucleotide Transformer embeddings
│   ├── fmhyena/                # Hyena embeddings
│   └── fmevo/                  # EVO2 embeddings
├── fusion-classifiers/         # Classification models
│   ├── train_embeddings.py     # Embedding-based classification
│   ├── train_fusionai.py       # FusionAI reimplementation
│   └── train_script_example.sh # Batch training script
└── notebooks/                  # Analysis notebooks
    └── compare_models.ipynb    # Results comparison
```

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/))

### Setup Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/kbi-fbmi/articles--2025expiratory.git
   cd fmfusions
   ```

2. **Open in VS Code** (recommended)
   ```bash
   code fmfusions.code-workspace
   ```
   Install recommended extensions when prompted.

3. **Initialize each project**
   ```bash
   # Install Python version and dependencies
   uv sync --python 3.11
   ```

4. **Add packages** (for development)
   ```bash
   # Add runtime dependency
   uv add <package>
   
   # Add development dependency
   uv add --dev <package>
   ```

## Usage

### Extract Embeddings

Each extractor module has its own `run.sh` script. Edit configuration as needed:

```bash
cd fmextractors/fmdnabert
bash run.sh

cd ../fmnt
bash run.sh

# ... repeat for other extractors
```

### Train Classification Models

```bash
cd fusion-classifiers

# Train single model
python train_embeddings.py

# Download data and run models
bash train_script_example.sh
```

### Analyze Results

```bash
cd notebooks
jupyter notebook compare_models.ipynb
```

## Dependencies

### Core Dependencies (fmlib)
- biopython: Sequence parsing and handling
- keras: Deep learning framework
- scikit-learn: Machine learning utilities
- torch: PyTorch tensors and models
- pandas: Data manipulation
- numpy: Numerical computing

### Classification Dependencies (fusion-classifiers)
- tensorflow/tf-keras: Deep learning
- matplotlib: Visualization
- jupyter: Interactive notebooks


## Data Resources

- Original FusionAI datasets: [FusionGDB2](https://compbio.uth.edu/FusionGDB2/)
- Precomputed embeddings and results: [Zenodo](https://zenodo.org/records/17898581)


## License

See LICENSE file for details.

## Contributing

Contributions are welcome. Please ensure code follows the project's style guidelines and includes appropriate tests.

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

## Published

Not yet

