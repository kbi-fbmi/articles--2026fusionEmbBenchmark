# Notebooks - Analysis and Visualization

This directory contains Jupyter notebooks for the article "Benchmarking Genomic Foundation Models for Gene Fusion Detection".

## Overview

The notebooks provide comprehensive analysis including:
- Loading precomputed embeddings from multiple foundation models
- Visualizing embedding spaces using t-SNE
- Comparing model performance metrics
- Evaluating sample efficiency across different models
- Training history analysis and learning curves

## Contents

### Main Notebook

- **[compare_models.ipynb](compare_models.ipynb)**: Primary analysis notebook that compares all benchmarked models
  - Downloads precomputed embeddings and trained models from Zenodo
  - Performs t-SNE visualization of embedding spaces
  - Analyzes classification performance
  - Compares sample efficiency and learning curves
  - Generates publication-ready figures

### Utilities

- **[helpers.py](helpers.py)**: Helper functions for data downloading and preprocessing
  - `zenodo_get_and_unzip()`: Download and extract datasets from Zenodo

### Data Directory

- **download/**: Contains downloaded data and results
  - `embeddings/`: Precomputed embeddings from all foundation models
  - `learned_models_nn/`: Trained neural network models
  - `learned_models_svm/`: Trained SVM models

## Quick Start

### Prerequisites

- Python 3.11+
- Jupyter Notebook or JupyterLab

### Setup

1. **Initialize the environment**
   ```bash
   cd notebooks
   uv sync --python 3.11
   ```

2. **Activate the environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

4. **Open and run** [compare_models.ipynb](compare_models.ipynb)
   - The notebook will automatically download required data from Zenodo on first run
   - All cells can be executed sequentially

## Data Sources

The notebooks automatically download data from Zenodo:
- **Zenodo Record**: [17898581](https://zenodo.org/records/17898581)
- **Embeddings**: Precomputed features from DNABERT, NT, Hyena, and EVO2
- **Trained Models**: Pre-trained neural network and SVM classifiers

## Key Analyses

### 1. Embedding Space Visualization
- t-SNE dimensionality reduction of embeddings
- Visual comparison of how different models separate fusion vs. non-fusion sequences

### 2. Model Performance Comparison
- Classification accuracy across all models
- Precision, recall, and F1-scores
- Confusion matrices

### 3. Sample Efficiency Analysis
- Learning curves showing performance vs. training set size
- Comparison of data efficiency across models
- Estimation of samples needed to reach target performance

### 4. Training Dynamics
- Loss curves and convergence analysis
- Comparison of training stability across models

## Dependencies

Key packages (automatically installed via `uv sync`):
- **fmlib**: Core library for fusion gene analysis
- **jupyter**: Interactive notebook environment
- **matplotlib**: Visualization and plotting
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities (t-SNE, metrics)
- **zenodo-get**: Automated dataset downloading
- **torch**: PyTorch for tensor operations

## Output

The notebook generates:
- Interactive visualizations
- Performance comparison tables
- Optional: Saved figures (set `save_images = True` in the notebook)

## Customization

To modify the analysis:
1. Edit `save_images` flag to save figures automatically
2. Adjust `sample_size` for different data subsets
3. Modify visualization parameters (colors, sizes, etc.)
4. Add custom analysis cells as needed

## Troubleshooting

### Data Download Issues
If automatic download fails:
```python
# Manually download from Zenodo
from helpers import zenodo_get_and_unzip
zenodo_get_and_unzip('17898581', 'embeddings.zip', 'download')
```

### Memory Issues
For large datasets, consider:
- Running on a machine with more RAM
- Processing models sequentially rather than loading all at once
- Reducing sample sizes for initial exploration

## Related Documentation

- [Main Project README](../README.md)
- [fmlib Documentation](../fmlib/README.md)
- [Fusion Classifiers](../fusion-classifiers/README.md)

## Citation
- not yet
