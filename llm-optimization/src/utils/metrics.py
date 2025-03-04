"""
Utilities for measuring model performance metrics
"""
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def measure_performance(model, eval_dataloader, device, prepare_batch_fn):
    """
    Measure baseline performance metrics for the model
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        prepare_batch_fn: Function to prepare batch for model input
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    model.to(device)
    model.eval()
    
    # Measure model size (in MB)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Measure inference metrics
    latencies = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            model_inputs = prepare_batch_fn(batch, device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(**model_inputs)
            inference_time = time.time() - start_time
            latencies.append(inference_time)
            
            # Compute accuracy
            predictions = outputs.logits.argmax(dim=-1)
            labels = model_inputs["labels"]
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_latency = sum(latencies) / len(latencies)
    accuracy = correct_predictions / total_predictions
    
    return {
        "model_size_mb": model_size_mb,
        "avg_latency_seconds": avg_latency,
        "avg_latency_ms": avg_latency * 1000,  # Convert to milliseconds
        "accuracy": accuracy,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "memory_footprint_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
    }


def save_metrics(metrics, file_path="baseline_metrics.pt"):
    """
    Save performance metrics to a file
    
    Args:
        metrics (dict): Dictionary of metrics to save
        file_path (str): Path to save metrics to
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    torch.save(metrics, file_path)
    print(f"Metrics saved to {file_path}")


def print_metrics(metrics, model_name=None, dataset_info=None):
    """
    Print performance metrics in a formatted way
    
    Args:
        metrics (dict): Dictionary of metrics
        model_name (str, optional): Name of the model
        dataset_info (str, optional): Information about the dataset
    """
    print("\n" + "="*50)
    if model_name:
        print(f"Model: {model_name}")
    if dataset_info:
        print(f"Task: {dataset_info}")
    print(f"Number of parameters: {metrics['num_parameters']:,}")
    print(f"Model size: {metrics['model_size_mb']:.2f} MB")
    print(f"Average inference latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    if metrics.get('memory_footprint_mb') is not None:
        print(f"GPU memory footprint: {metrics['memory_footprint_mb']:.2f} MB")
    print("="*50)


def plot_metrics(metrics_list, names, save_path=None):
    """
    Plot comparison of metrics for different models
    
    Args:
        metrics_list (list): List of metrics dictionaries
        names (list): List of model names
        save_path (str, optional): Path to save plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Size plot
    sizes = [m["model_size_mb"] for m in metrics_list]
    axs[0, 0].bar(names, sizes)
    axs[0, 0].set_title("Model Size (MB)")
    axs[0, 0].set_ylabel("MB")
    
    # Latency plot
    latencies = [m["avg_latency_ms"] for m in metrics_list]
    axs[0, 1].bar(names, latencies)
    axs[0, 1].set_title("Inference Latency (ms)")
    axs[0, 1].set_ylabel("ms")
    
    # Accuracy plot
    accuracies = [m["accuracy"] for m in metrics_list]
    axs[1, 0].bar(names, accuracies)
    axs[1, 0].set_title("Accuracy")
    axs[1, 0].set_ylim(0, 1)
    
    # Parameters plot
    parameters = [m["num_parameters"] / 1_000_000 for m in metrics_list]
    axs[1, 1].bar(names, parameters)
    axs[1, 1].set_title("Number of Parameters (M)")
    axs[1, 1].set_ylabel("Millions")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
