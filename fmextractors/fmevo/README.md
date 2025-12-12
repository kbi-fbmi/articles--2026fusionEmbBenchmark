# Embeddings computation for EVO2 
This implementation uses the [EVO2 model](https://github.com/ArcInstitute/evo2.git).
## Installation

## Hardware and software Requirements

CUDA: 12.1+ with compatible NVIDIA drivers
cuDNN: 9.3+
Compiler: GCC 9+ or Clang 10+ with C++17 support
Python 3.11.11 required

Install dependencies using uv with Python 3.11.11:
```bash
bash init_evo2.sh
```

## Usage

Edit `run.sh` with your configuration, then execute:

```bash
bash run-evo2.sh
```

## Overview

This module computes embeddings for DNA sequences using Evo2 models. Configure the script parameters before running to specify input data, model selection, and output paths.