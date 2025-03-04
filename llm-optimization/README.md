# LLM Optimization for Edge Devices

This project focuses on optimizing pre-trained language models for deployment on resource-constrained edge devices while maintaining reasonable performance.

## Overview

The goal is to reduce the model size by at least 50% while maintaining at least 90% of its original performance using techniques such as:
- Knowledge distillation
- Pruning
- Quantization
- Inference optimization

## Project Structure

- `src/`: Source code for model implementations and utilities
  - `data/`: Data loading and processing utilities
  - `models/`: Model implementation and configuration
  - `utils/`: Utility functions for metrics and evaluation
- `notebooks/`: Jupyter notebooks for experiments and demonstrations

## Getting Started

1. Install the requirements:
```
pip install -r requirements.txt
```

## Baseline Model

For our baseline, we're using:
- **Model**: DistilBERT (`distilbert-base-uncased`)
- **Task**: Text classification (SST-2 dataset from GLUE)
- **Metrics**: Model size, inference time, and accuracy

## Next Steps

After establishing the baseline, we'll implement optimization techniques including:
1. Pruning
2. Quantization
3. Model compilation
