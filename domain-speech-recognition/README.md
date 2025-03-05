# Domain-Specific Speech Recognition - Dataset Creation

This module creates a synthetic dataset for fine-tuning a speech recognition model for law enforcement terminology in noisy environments.

## Setup

1. Install the requirements:

pip install -r requirements.txt

2. Add noise sample files to the `data/noise_samples/` directory:
- `wind_noise.wav`
- `traffic_noise.wav`
- `crowd_noise.wav`

You can download sample noise files from [FreeSound.org](https://freesound.org/) or similar sources.

## Creating the Dataset

Run the dataset creation script:

python src/dataset_creation.py

This will:
1. Generate clean speech samples for each phrase
2. Create noisy versions with different noise types and SNR levels
3. Save all audio files to the `data/generated/` directory
4. Generate train/validation splits
5. Optionally convert to HuggingFace Dataset format

You can validate dataset:

python src/validate_dataset.py 

## Dataset Structure

The generated dataset includes:
- Audio files for each sample (clean and with different noise types)
- Metadata JSON files with transcriptions and audio properties
- Waveform visualizations

## Configuration

You can modify the dataset parameters in `src/config.py`:
- Add or modify police phrases
- Change noise types and SNR levels
- Adjust output directories and sampling rates

## Next Steps

After dataset creation, proceed to:
1. Fine-tune the Whisper model using LoRA
2. Implement the audio preprocessing pipeline
3. Build the evaluation framework
