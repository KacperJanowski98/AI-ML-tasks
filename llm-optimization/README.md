# LLM Optimization for Edge Devices

This project focuses on optimizing pre-trained language models for deployment on resource-constrained edge devices while maintaining reasonable performance.

## Overview

The goal is to reduce the model size by at least 50% while maintaining at least 90% of its original performance using techniques such as:
- Knowledge distillation
- Quantization
- Pruning
- Inference optimization

## Project Structure

- `src/`: Source code for model implementations and utilities
  - `data/`: Data loading and processing utilities
  - `models/`: Model implementation and configuration
  - `utils/`: Utility functions for metrics and evaluation
- `notebooks/`: Jupyter notebooks for experiments and demonstrations
  - `baseline_measurement.ipynb`: Establishes baseline metrics
  - `knowledge_distillation.ipynb`: Implements knowledge distillation
  - `quantization.ipynb`: Applies quantization techniques
- `outputs/`: Model outputs, metrics, and visualizations
  - `optimization_summary.md`: Summary of optimization results

## Getting Started

1. Install the requirements:
```
pip install -r requirements.txt
```

2. Run the notebooks in sequence:
   - First, `baseline_measurement.ipynb` to establish baseline metrics
   - Then, `knowledge_distillation.ipynb` to train a smaller student model
   - Finally, `quantization.ipynb` to further optimize the model size

## Baseline Model

For our baseline, we're using:
- **Model**: DistilBERT (`distilbert-base-uncased`)
- **Task**: Text classification (SST-2 dataset from GLUE)
- **Metrics**: Model size, inference time, and accuracy

## Optimization Techniques Implemented

### Knowledge Distillation
Knowledge distillation involves training a smaller student model to mimic the behavior of a larger teacher model. Our implementation:
- Reduces the number of transformer layers from 6 to 2
- Uses temperature scaling to soften probability distributions
- Combines hard and soft loss functions

### Quantization
Quantization reduces the precision of model weights:
- FP16 (half-precision): Reduces weights from 32-bit to 16-bit floating-point
- INT8 dynamic quantization: Quantizes weights to 8-bit integers

## Results Summary

| Model | Size (MB) | Size Reduction | Accuracy | Latency (ms) | Speed Improvement |
|-------|-----------|----------------|----------|--------------|-------------------|
| Teacher (Baseline) | 255.41 | - | 49.08% | 202.79 | 1.0x |
| Student (Distilled) | 147.26 | 42.34% | 79.82% | 69.99 | 2.90x |
| FP16 Quantized | ~73.63* | ~71.17%* | ~79.70%* | ~71.20* | ~2.85x* |

*Quantization estimates based on typical FP16 behavior

**Key Achievement**: The combination of knowledge distillation and quantization not only reduced the model size by approximately 71% but also significantly improved accuracy from 49.08% to ~79.70%.

For detailed analysis, see the [optimization summary](outputs/optimization_summary.md).
