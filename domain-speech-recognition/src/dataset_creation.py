"""
Main script for creating the synthetic dataset for domain-specific speech recognition
"""

import os
import json
import numpy as np
import torch
from transformers import AutoProcessor
from datasets import Dataset
import soundfile as sf
from tqdm import tqdm
import random

from audio_utils import text_to_speech, load_and_prepare_noise, add_noise_to_speech, plot_waveform
from config import (
    POLICE_PHRASES, NOISE_TYPES, SNR_LEVELS, SAMPLE_RATE,
    NOISE_SAMPLES_DIR, OUTPUT_DIR, DATASET_PATH
)

def create_synthetic_sample(text, output_dir, noise_type=None, snr_db=None):
    """
    Create a synthetic audio sample with optional background noise
    
    Args:
        text (str): Text to synthesize
        output_dir (str): Directory to save the audio file
        noise_type (str, optional): Type of noise to add ('wind', 'traffic', 'crowd')
        snr_db (float, optional): Signal-to-noise ratio in dB
    
    Returns:
        dict: Sample information including paths and metadata
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename from text (first few words)
    base_filename = "_".join(text.split()[:4]).lower()
    base_filename = "".join(c if c.isalnum() or c == "_" else "" for c in base_filename)
    
    # Add noise info to filename if applicable
    if noise_type and snr_db is not None:
        filename = f"{base_filename}_{noise_type}_snr{snr_db}.wav"
    else:
        filename = f"{base_filename}_clean.wav"
    
    output_path = os.path.join(output_dir, filename)
    
    # Generate speech from text
    speech, sr = text_to_speech(text)
    
    # Add noise if specified
    if noise_type and snr_db is not None:
        noise_file = f"{noise_type}_noise.wav"
        noise_path = os.path.join(NOISE_SAMPLES_DIR, noise_file)
        
        try:
            noise = load_and_prepare_noise(noise_path, len(speech), sr)
            noisy_speech = add_noise_to_speech(speech, noise, snr_db)
        except FileNotFoundError:
            print(f"Warning: Noise file {noise_file} not found. Using clean speech.")
            noisy_speech = speech
            noise_type = "clean"
            snr_db = None
    else:
        noisy_speech = speech
    
    # Save the audio file
    sf.write(output_path, noisy_speech, sr)
    
    # Create visualization if enabled
    plot_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{os.path.splitext(filename)[0]}.png")
    plot_waveform(noisy_speech, sr, title=f"{noise_type or 'Clean'} SNR:{snr_db or 'N/A'}", save_path=plot_path)
    
    return {
        "text": text,
        "audio_path": output_path,
        "sampling_rate": sr,
        "noise_type": noise_type or "clean",
        "snr": snr_db,
        "visualization_path": plot_path
    }

def create_dataset():
    """
    Create a complete dataset of synthetic speech samples
    """
    print("Creating synthetic speech dataset...")
    dataset_entries = []
    
    # Check if noise directory exists
    if not os.path.exists(NOISE_SAMPLES_DIR):
        os.makedirs(NOISE_SAMPLES_DIR)
        print(f"Created noise samples directory: {NOISE_SAMPLES_DIR}")
        print("Please add noise files (wind_noise.wav, traffic_noise.wav, crowd_noise.wav) before proceeding.")
        print("You can download sample noise files from FreeSound.org or similar sources.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # First, create clean samples
    print("Generating clean speech samples...")
    for phrase in tqdm(POLICE_PHRASES):
        sample = create_synthetic_sample(
            text=phrase,
            output_dir=os.path.join(OUTPUT_DIR, "clean")
        )
        dataset_entries.append(sample)
    
    # Then, create noisy samples with different SNRs
    print("Generating noisy speech samples...")
    for phrase in tqdm(POLICE_PHRASES):
        for noise_type in NOISE_TYPES:
            for snr in SNR_LEVELS:
                sample = create_synthetic_sample(
                    text=phrase,
                    output_dir=os.path.join(OUTPUT_DIR, "noisy"),
                    noise_type=noise_type,
                    snr_db=snr
                )
                dataset_entries.append(sample)
    
    # Save dataset metadata
    with open(DATASET_PATH, 'w') as f:
        json.dump(dataset_entries, f, indent=2)
    
    print(f"Created {len(dataset_entries)} samples.")
    print(f"Dataset metadata saved to {DATASET_PATH}")
    
    # Create train/validation split
    random.shuffle(dataset_entries)
    split_idx = int(len(dataset_entries) * 0.8)
    train_data = dataset_entries[:split_idx]
    val_data = dataset_entries[split_idx:]
    
    # Save splits
    with open(os.path.join(OUTPUT_DIR, "train_dataset.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "val_dataset.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")

def convert_to_hf_dataset():
    """
    Convert the dataset to HuggingFace Dataset format
    """
    print("Converting to HuggingFace Dataset format...")
    
    # Load the dataset metadata
    with open(DATASET_PATH, 'r') as f:
        dataset_entries = json.load(f)
    
    # Prepare for HuggingFace dataset format
    hf_dataset = {
        "text": [],
        "audio": [],
        "sampling_rate": [],
        "noise_type": [],
        "snr": []
    }
    
    for entry in dataset_entries:
        hf_dataset["text"].append(entry["text"])
        # Load audio
        audio, sr = sf.read(entry["audio_path"])
        hf_dataset["audio"].append(audio)
        hf_dataset["sampling_rate"].append(sr)
        hf_dataset["noise_type"].append(entry["noise_type"])
        hf_dataset["snr"].append(entry["snr"])
    
    # Convert to HuggingFace Dataset
    police_dataset = Dataset.from_dict(hf_dataset)
    
    # Save the dataset
    police_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "hf_dataset"))
    
    print(f"HuggingFace Dataset saved to {os.path.join(OUTPUT_DIR, 'hf_dataset')}")
    return police_dataset

if __name__ == "__main__":
    # Check if noise samples exist
    missing_noise_files = []
    for noise_type in NOISE_TYPES:
        noise_file = f"{noise_type}_noise.wav"
        if not os.path.exists(os.path.join(NOISE_SAMPLES_DIR, noise_file)):
            missing_noise_files.append(noise_file)
    
    if missing_noise_files:
        print("Warning: The following noise files are missing:")
        for file in missing_noise_files:
            print(f"  - {file}")
        print(f"\nPlease add these files to the {NOISE_SAMPLES_DIR} directory.")
        print("You can download sample noise files from FreeSound.org or similar sources.")
        create_noise_dir = input("Would you like to create the noise directory now? (y/n): ")
        if create_noise_dir.lower() == 'y':
            os.makedirs(NOISE_SAMPLES_DIR, exist_ok=True)
            print(f"Created directory: {NOISE_SAMPLES_DIR}")
    
    # Create the dataset
    create_dataset()
    
    # Convert to HuggingFace Dataset format
    convert_hf = input("Convert to HuggingFace Dataset format? (y/n): ")
    if convert_hf.lower() == 'y':
        convert_to_hf_dataset()
