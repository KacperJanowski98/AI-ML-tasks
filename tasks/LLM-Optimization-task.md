# LLM Optimization for Edge Devices

## Overview

This task focuses on optimizing a pre-trained language model for deployment on resource-constrained edge devices while maintaining reasonable performance. You'll explore and implement techniques such as knowledge distillation, pruning, quantization, and inference optimization.

## Objectives

1. Optimize a pre-trained language model (e.g., BERT, GPT-2 Small, or LLaMA-7B) for edge deployment
2. Reduce the model size by at least 50% while maintaining at least 90% of its original performance
3. Implement and compare at least two different optimization techniques
4. Document your approach, challenges, and trade-offs

## Task Requirements

### 1. Model Selection and Baseline Measurement

- Choose a reasonably sized pre-trained language model (e.g., BERT-base, DistilBERT, GPT-2 Small, or LLaMA-7B)
- Select a relevant evaluation task (text classification, sentiment analysis, question answering, etc.)
- Establish baseline performance metrics (accuracy, inference time, model size)

```python
# Example code structure for baseline measurements
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

def measure_performance(model, tokenizer, dataset, device):
    model.to(device)
    model.eval()
    
    # Measure model size
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Assuming float32
    
    # Measure inference time
    latencies = []
    accuracies = []
    
    with torch.no_grad():
        for batch in dataset:
            start_time = time.time()
            outputs = model(**batch)
            latencies.append(time.time() - start_time)
            
            # Compute accuracy
            predictions = outputs.logits.argmax(dim=-1)
            accuracies.append((predictions == batch["labels"]).float().mean().item())
    
    avg_latency = sum(latencies) / len(latencies)
    avg_accuracy = sum(accuracies) / len(accuracies)
    
    return {
        "model_size_mb": model_size_mb,
        "avg_latency_seconds": avg_latency,
        "accuracy": avg_accuracy
    }
```

### 2. Optimization Techniques (Implement at least two)

#### Knowledge Distillation

Train a smaller student model that mimics the behavior of the larger teacher model:

```python
class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=2.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.5):
        """
        Compute the distillation loss: a weighted average of the hard cross-entropy loss
        and the soft distillation loss (KL-divergence between soft student and teacher outputs)
        """
        import torch.nn.functional as F
        
        # Hard loss: standard cross-entropy with true labels
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss: KL divergence between soft student and teacher outputs
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
    def train(self, dataloader, optimizer, device, epochs=3):
        # Training loop implementation
        pass
```

#### Pruning

Remove less important weights or neurons from the model:

```python
def magnitude_based_pruning(model, pruning_ratio=0.3):
    """
    Simple magnitude-based pruning: set the smallest weights to zero
    """
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only prune weights, not biases
            tensor = param.data.cpu().numpy()
            threshold = np.percentile(np.abs(tensor), pruning_ratio * 100)
            mask = np.abs(tensor) > threshold
            param.data = torch.from_numpy(tensor * mask).to(param.device)
    
    return model

def structured_pruning(model, pruning_ratio=0.3):
    """
    Structured pruning: remove entire neurons/filters
    """
    # Implementation of structured pruning
    pass
```

#### Quantization

Reduce the precision of the weights and/or activations:

```python
def quantize_model(model, quantization_config):
    """
    Apply post-training quantization to the model
    """
    import torch.quantization
    
    # Prepare the model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate the model
    # ... run representative data through the model ...
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model
```

#### Other Optimization Methods

- Model compilation (e.g., using `torch.compile()` for PyTorch 2.0+)
- ONNX Runtime or TensorRT conversion
- Layer fusion and operator optimization

### 3. Evaluation and Comparison

Create a comprehensive evaluation framework to compare the optimized models:

```python
def compare_models(original_model, optimized_models, test_dataset, device):
    """
    Compare the original model with optimized versions
    """
    results = {}
    
    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = measure_performance(original_model, tokenizer, test_dataset, device)
    results["original"] = original_metrics
    
    # Evaluate each optimized model
    for name, model in optimized_models.items():
        print(f"Evaluating {name}...")
        metrics = measure_performance(model, tokenizer, test_dataset, device)
        
        # Calculate relative performance
        rel_metrics = {
            "size_reduction": 1 - (metrics["model_size_mb"] / original_metrics["model_size_mb"]),
            "speedup": original_metrics["avg_latency_seconds"] / metrics["avg_latency_seconds"],
            "accuracy_retention": metrics["accuracy"] / original_metrics["accuracy"]
        }
        
        results[name] = {**metrics, **rel_metrics}
    
    return results
```

### 4. Edge Deployment Considerations

Document how your optimized model would be deployed to edge devices:

- Memory constraints and loading strategies
- Inference pipeline optimization
- Handling variable inputs
- Batch processing (if applicable)

## Deliverables

1. **Code Repository**: Jupyter notebooks or Python scripts implementing:
   - Baseline model evaluation
   - At least two optimization techniques
   - Evaluation framework

2. **Performance Report**: A markdown document containing:
   - Comparison of optimization techniques (size reduction, speed improvement, accuracy impact)
   - Trade-offs between different approaches
   - Visualizations of performance metrics

3. **Deployment Guide**: Brief documentation on how to:
   - Export your optimized model
   - Load and run it on an edge device
   - Expected real-world performance characteristics

## Evaluation Criteria

Your solution will be evaluated based on:

1. **Effectiveness of Optimization**: How well you reduced model size and improved inference speed while maintaining accuracy
2. **Technical Implementation**: Quality of implementation and code organization
3. **Comparison Depth**: How thoroughly you analyzed the trade-offs between different optimization approaches
4. **Documentation Quality**: Clarity of your performance report and deployment guide

## Resources and Hints

- Start with a reasonably sized model that fits your available compute resources
- Consider using the Hugging Face Transformers library for easy access to pre-trained models
- For testing without actual edge hardware, you can simulate constraints (e.g., limit CPU cores, available RAM)
- Quantization-aware training often outperforms post-training quantization for maintaining accuracy

## Time Expectation

- This task should take approximately 3-5 hours for an experienced ML engineer
- If time is limited, focus on implementing one optimization technique thoroughly rather than multiple techniques superficially
