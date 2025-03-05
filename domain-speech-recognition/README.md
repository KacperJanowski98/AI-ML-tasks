# Domain-Specific Speech Recognition

This project implements a fine-tuned Whisper model for domain-specific speech recognition in challenging environments, specifically for law enforcement terminology in noisy conditions.

## Overview

Speech recognition systems often struggle with specialized terminology and noisy environments. This project addresses these challenges by fine-tuning a pre-trained Whisper model using Low-Rank Adaptation (LoRA) on a synthetic dataset containing law enforcement terminology mixed with various types of background noise.

## Features

- **Synthetic Dataset Creation**: Generate speech samples with domain-specific terminology and controllable noise conditions
- **LoRA Fine-Tuning**: Efficient fine-tuning of Whisper model with minimal trainable parameters (~1.4%)
- **Audio Processing Pipeline**: Preprocessing for handling noisy audio inputs
- **Evaluation Framework**: Assessment of model performance with Word Error Rate (WER) metrics
- **Inference Tools**: Easy-to-use script for transcribing new audio files

## Project Structure

domain-speech-recognition/
├── data/
│   ├── noise_samples/         # Background noise audio files
│   │   ├── crowd_noise.wav
│   │   ├── traffic_noise.wav
│   │   └── wind_noise.wav
│   ├── generated/             # Generated synthetic dataset
│   └── processed_dataset/     # HuggingFace dataset for training
├── models/
│   └── whisper-lora-simple/   # Fine-tuned model outputs
├── src/
│   ├── audio_utils.py         # Audio processing utilities
│   ├── config.py              # Configuration parameters
│   ├── dataset_creation.py    # Synthetic dataset generation
│   ├── create_whisper_dataset.py  # Dataset preparation for training
│   ├── train_whisper.py       # Model fine-tuning
│   ├── inference.py           # Inference script for model testing
│   └── validate_dataset.py    # Dataset validation
└── requirements.txt           # Project dependencies

## Setup

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Verify noise sample files are present in the data/noise_samples/ directory:

- wind_noise.wav
- traffic_noise.wav
- crowd_noise.wav

## Usage

1. Dataset Creation

Generate the synthetic dataset with domain-specific phrases in various noise conditions:

```bash
python src/dataset_creation.py
```

This will:

- Generate speech samples for each phrase in the configuration
- Create noisy versions with different noise types and SNR levels
- Save all files to the data/generated/ directory
- Create train/validation splits

2. Prepare Dataset for Training

Convert the generated data to a format suitable for Whisper fine-tuning:

```bash
python src/create_whisper_dataset.py
```

3. Train the Model

Fine-tune Whisper using LoRA:

```bash
python src/train_whisper.py
```

4. Inference

Transcribe audio files using the fine-tuned model:

```bash
# Transcribe a single file
python src/inference.py --file path/to/audio.wav

# Transcribe all files in a directory
python src/inference.py --dir path/to/directory

# Create visualizations along with transcriptions
python src/inference.py --file path/to/audio.wav --visualize
```

## Results

The fine-tuned model achieves a Word Error Rate (WER) of approximately 11.3% on domain-specific speech in noisy environments, significantly better than the baseline Whisper model performance on specialized terminology.

Performance characteristics:

- Training parameters: 3.5M trainable parameters (1.44% of base model)
- Audio processing: Successfully handles various noise conditions
- Domain adaptation: Improved recognition of law enforcement terminology

## Future Improvements

Potential next steps for enhancing the model:

1. Expand the synthetic dataset with more diverse phrases and noise conditions
2. Experiment with different LoRA configurations for potentially better performance
3. Implement real-time streaming for practical applications
4. Create a simple UI for testing the model
5. Add confidence scoring to flag potential transcription errors

## Notes

- The model is optimized for short audio clips (up to 30 seconds)
- Best performance is achieved on clear speech with moderate background noise
- For very noisy environments, the audio preprocessing pipeline can be further optimized
