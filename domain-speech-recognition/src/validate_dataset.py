import os
import json
import torch
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import WhisperProcessor, WhisperFeatureExtractor
import soundfile as sf
import matplotlib.pyplot as plt

# Path to your dataset
DATASET_PATH = "data/generated/train_dataset.json"
OUTPUT_DIR = "data/validation_results"
BASE_MODEL_ID = "openai/whisper-small"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset():
    """Load the dataset and return basic info"""
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset loaded with {len(data)} entries")
    
    # Print sample entry
    print("\nSample entry:")
    print(json.dumps(data[0], indent=2))
    
    return data

def check_audio_files(data):
    """Check if all audio files exist and can be loaded"""
    print("\nChecking audio files...")
    
    missing_files = []
    invalid_files = []
    
    for i, entry in enumerate(data):
        audio_path = entry.get("audio_path")
        
        if not audio_path:
            print(f"Entry {i} has no audio_path field")
            continue
            
        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue
            
        try:
            # Try to load the audio file
            audio, sr = sf.read(audio_path)
            
            # Check if audio is valid
            if len(audio) == 0:
                invalid_files.append(audio_path)
                continue
                
            # Print details for first few files
            if i < 5:
                print(f"File {i}: {os.path.basename(audio_path)}")
                print(f"  - Duration: {len(audio)/sr:.2f}s")
                print(f"  - Sample rate: {sr}Hz")
                print(f"  - Shape: {audio.shape}")
                print(f"  - Min/Max values: {audio.min():.2f}/{audio.max():.2f}")
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            invalid_files.append(audio_path)
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} missing audio files")
        for path in missing_files[:5]:  # Print first 5 only
            print(f"  - {path}")
        if len(missing_files) > 5:
            print(f"  - ...and {len(missing_files)-5} more")
    
    if invalid_files:
        print(f"\nWARNING: {len(invalid_files)} invalid audio files")
        for path in invalid_files[:5]:
            print(f"  - {path}")
        if len(invalid_files) > 5:
            print(f"  - ...and {len(invalid_files)-5} more")
            
    if not missing_files and not invalid_files:
        print("All audio files exist and are valid!")
        
    return len(missing_files) == 0 and len(invalid_files) == 0

def test_feature_extraction():
    """Test feature extraction with Whisper processor"""
    print("\nTesting Whisper feature extraction...")
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
    
    # Load a sample audio file
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No entries in dataset!")
        return False
        
    sample_entry = data[0]
    audio_path = sample_entry.get("audio_path")
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Sample audio file not found: {audio_path}")
        return False
    
    try:
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Extract features with feature extractor directly
        features = feature_extractor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        
        print(f"Feature extraction successful!")
        print(f"Input features shape: {features.input_features.shape}")
        
        # Try with processor too
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        print(f"Processor successful!")
        print(f"Processor keys: {inputs.keys()}")
        
        # Save visualizations
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(audio))/sr, audio)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(os.path.join(OUTPUT_DIR, "sample_waveform.png"))
        
        # Visualize features (mel spectrogram)
        plt.figure(figsize=(10, 4))
        plt.imshow(features.input_features[0].numpy(), aspect='auto', origin='lower')
        plt.title('Mel Spectrogram Features')
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(os.path.join(OUTPUT_DIR, "sample_features.png"))
        
        return True
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        return False

def test_dataset_pipeline():
    """Test the full dataset pipeline as used in training"""
    print("\nTesting full dataset pipeline...")
    
    try:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
        
        # Load data
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
            
        # Create a small sample for testing
        sample_data = data[:5]
        
        # Format for HuggingFace Dataset
        dataset_dict = {
            "text": [item["text"] for item in sample_data],
            "audio": [item["audio_path"] for item in sample_data],
            "noise_type": [item["noise_type"] for item in sample_data],
            "snr": [item["snr"] for item in sample_data]
        }
        
        # Create dataset
        from datasets import Dataset, Audio
        test_dataset = Dataset.from_dict(dataset_dict)
        
        # Cast to audio
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print("Dataset created successfully with audio column")
        print(f"Dataset features: {test_dataset.features}")
        
        # Define preprocessing function
        def preprocess_function(examples):
            # Load audio
            audio = [x["array"] for x in examples["audio"]]
            
            # Extract features
            features = processor.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="np", 
                padding=True
            )
            
            # Tokenize text
            labels = processor.tokenizer(examples["text"], padding=True).input_ids
            
            # Return as expected for Whisper
            return {
                "input_features": features.input_features.tolist(),
                "labels": labels
            }
        
        # Apply preprocessing
        processed_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
        )
        
        print("\nProcessed dataset:")
        print(f"Features: {processed_dataset.features}")
        print(f"First example keys: {list(processed_dataset[0].keys())}")
        print(f"Input features type: {type(processed_dataset[0]['input_features'])}")
        print(f"Input features length: {len(processed_dataset[0]['input_features'])}")
        
        # Create a small batch for testing - USING CORRECT DATASET ACCESS
        batch = {
            "input_features": torch.tensor([processed_dataset[i]["input_features"] for i in range(2)]),
            "labels": torch.tensor([processed_dataset[i]["labels"] for i in range(2)])
        }
        
        print("\nBatch created successfully!")
        print(f"Batch keys: {batch.keys()}")
        print(f"input_features shape: {batch['input_features'].shape}")
        
        return True
    except Exception as e:
        print(f"Error in dataset pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Dataset Validation ===")
    data = load_dataset()
    
    audio_valid = check_audio_files(data)
    features_valid = test_feature_extraction()
    pipeline_valid = test_dataset_pipeline()
    
    print("\n=== Validation Summary ===")
    print(f"Audio files valid: {'✅' if audio_valid else '❌'}")
    print(f"Feature extraction valid: {'✅' if features_valid else '❌'}")
    print(f"Dataset pipeline valid: {'✅' if pipeline_valid else '❌'}")
    
    if audio_valid and features_valid and pipeline_valid:
        print("\n✅ Dataset appears to be valid and ready for training!")
    else:
        print("\n❌ Dataset has issues that need to be fixed!")
