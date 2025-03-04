from datasets import load_dataset
from torch.utils.data import DataLoader


def load_and_prepare_data(tokenizer, dataset_name="glue", dataset_config="sst2", 
                          batch_size=16, max_length=128):
    """
    Load and prepare dataset for text classification
    
    Args:
        tokenizer: The tokenizer to use for tokenization
        dataset_name (str): Name of the dataset
        dataset_config (str): Configuration of the dataset
        batch_size (int): Batch size for data loaders
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (tokenizer, train_dataloader, eval_dataloader)
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    # Create data loaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size
    )
    
    return tokenizer, train_dataloader, eval_dataloader


def prepare_batch_for_model(batch, device):
    """
    Convert dataset batch to model inputs on the correct device
    
    Args:
        batch: A batch from the dataloader
        device: Device to move tensors to
        
    Returns:
        dict: Batch dictionary ready for model input
    """
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device)
    }
