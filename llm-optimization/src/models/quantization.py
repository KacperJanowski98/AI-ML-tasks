"""
Quantization functionality for model optimization
"""
import os
import torch
import torch.nn as nn
import torch.quantization
from transformers import PreTrainedModel


def measure_model_size(model):
    """
    Measure the size of a model in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def apply_dynamic_quantization(model, quantization_dtype=torch.qint8):
    """
    Apply dynamic quantization to a model
    
    Args:
        model: PyTorch model
        quantization_dtype: Data type to quantize to (torch.qint8, torch.float16)
        
    Returns:
        model: Quantized model
        float: Size of the model in MB before quantization
        float: Size of the model in MB after quantization
    """
    # Measure size before quantization
    size_before = measure_model_size(model)
    
    # Clone the model to avoid modifying the original
    model_copy = type(model)(model.config) if hasattr(model, 'config') else type(model)()
    model_copy.load_state_dict(model.state_dict())
    
    # Get modules to quantize - find all Linear layers
    modules_to_quantize = []
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_quantize.append(name)
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},  # Specify which modules to quantize
        dtype=quantization_dtype
    )
    
    # Measure size after quantization
    size_after = measure_model_size(quantized_model)
    
    return quantized_model, size_before, size_after


def convert_to_fp16(model):
    """
    Convert a model to FP16 (half precision)
    
    Args:
        model: PyTorch model
        
    Returns:
        model: FP16 model
        float: Size of the model in MB before conversion
        float: Size of the model in MB after conversion
    """
    # Measure size before conversion
    size_before = measure_model_size(model)
    
    # Clone the model to avoid modifying the original
    model_copy = type(model)(model.config) if hasattr(model, 'config') else type(model)()
    model_copy.load_state_dict(model.state_dict())
    
    # Convert parameters to half precision
    for param in model_copy.parameters():
        param.data = param.data.half()
    
    # Convert buffers to half precision
    for buffer in model_copy.buffers():
        buffer.data = buffer.data.half()
    
    # Measure size after conversion
    size_after = measure_model_size(model_copy)
    
    return model_copy, size_before, size_after


def export_to_onnx(model, tokenizer, file_path, input_shape=(1, 128), opset_version=12):
    """
    Export a transformer model to ONNX format
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        file_path: Path to save ONNX model
        input_shape: Shape of input tensor
        opset_version: ONNX opset version
        
    Returns:
        str: Path to saved ONNX model
    """
    # Create dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, input_shape)
    attention_mask = torch.ones(input_shape)
    
    # Prepare inputs
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs,),
        file_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        do_constant_folding=True,
        opset_version=opset_version
    )
    
    # Get file size in MB
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    return file_path, size_mb


class QuantizedModelWrapper(nn.Module):
    """
    Wrapper class for quantized models to ensure compatibility with HuggingFace API
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Copy important attributes from the wrapped model
        if hasattr(model, 'config'):
            self.config = model.config
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def to(self, device):
        # Quantized models may not support moving to GPU
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU instead")
            device = 'cpu'
        
        try:
            self.model.to(device)
        except RuntimeError:
            print("Warning: Cannot move quantized model to GPU, keeping on CPU")
        
        return self


def save_quantized_model(model, path):
    """
    Save a quantized model
    
    Args:
        model: Quantized model
        path: Directory path to save the model
        
    Returns:
        str: Path where model was saved
    """
    os.makedirs(path, exist_ok=True)
    
    # Save the state dict
    torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))
    
    # Save config if available
    if hasattr(model, 'config'):
        model.config.save_pretrained(path)
    
    return path
