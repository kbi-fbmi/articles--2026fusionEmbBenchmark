# Embeddings computation for DNABERT 
This implementation uses the [DNABERT](https://huggingface.co/zhihan1996/DNABERT-2-117M).
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

This module computes embeddings for DNA sequences using DNABERT models. Configure the script parameters before running to specify input data, model selection, and output paths.