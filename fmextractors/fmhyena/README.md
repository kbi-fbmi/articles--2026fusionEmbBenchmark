# Embeddings computation for HyenaDNA 
This implementation uses the [HyenaDNA](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf).
## Installation
Install dependencies using uv with Python 3.11.11:

```bash
uv sync --python 3.11.11
```

## Usage

Edit `run.sh` with your configuration, then execute:

```bash
bash run.sh
```

## Overview

This module computes embeddings for DNA sequences using Nucleotide Transformer models. Configure the script parameters before running to specify input data, model selection, and output paths.