import os
import json
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Configuration
DATA_PATH = "data/generated"
TRAIN_JSON = "data/generated/train_dataset.json"
VAL_JSON = "data/generated/val_dataset.json"
OUTPUT_DIR = "data/processed_dataset"

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process training and validation data
    print("Processing training data...")
    train_data = process_json_data(TRAIN_JSON)
    
    print("Processing validation data...")
    val_data = process_json_data(VAL_JSON)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data)
    })
    
    # Save the processed dataset
    print(f"Saving processed dataset to {OUTPUT_DIR}")
    dataset_dict.save_to_disk(OUTPUT_DIR)
    
    print("Dataset creation complete!")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")

def process_json_data(json_path):
    """Process JSON data into simple format suitable for Whisper"""
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize dataset dict
    dataset = {
        "audio": [],
        "text": [],
    }
    
    # Process each sample
    for item in tqdm(data):
        try:
            # Add text
            dataset["text"].append(item["text"])
            
            # Add audio path
            dataset["audio"].append(item["audio_path"])
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    return dataset


if __name__ == "__main__":
    main()
