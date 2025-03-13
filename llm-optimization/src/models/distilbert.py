"""
DistilBERT model setup and configuration
"""
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


def load_distilbert_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Load a pre-trained DistilBERT model for sequence classification
    
    Args:
        model_name (str): Name or path of the pre-trained model
        num_labels (int): Number of classes for classification
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Load pre-trained model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    return model, tokenizer


def get_device():
    """
    Get the appropriate device (GPU if available, otherwise CPU)
    
    Returns:
        torch.device: The device to use
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
