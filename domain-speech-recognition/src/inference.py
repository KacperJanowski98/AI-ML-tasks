import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Configure model paths
BASE_MODEL = "openai/whisper-small"
LORA_PATH = "models/whisper-lora-simple/final"

def load_model():
    """Load the fine-tuned model with LoRA adapters"""
    print("Loading base model and processor...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    
    print(f"Loading LoRA adapters from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # Set model to evaluation mode
    model.eval()
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    return model, processor, device

def load_audio(file_path):
    """Load and preprocess audio file"""
    print(f"Loading audio: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load audio using soundfile
    audio, sample_rate = sf.read(file_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("Converting stereo to mono")
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}Hz")
    return audio, sample_rate

def transcribe_audio(audio_path, model, processor, device):
    """Transcribe a single audio file"""
    # Load and preprocess audio
    audio, sample_rate = load_audio(audio_path)
    
    # Convert to feature inputs 
    inputs = processor(
        audio, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).to(device)
    
    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_features=inputs.input_features,
            max_new_tokens=225
        )
    
    # Decode the generated ids
    transcription = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription

def batch_transcribe(directory, model, processor, device, file_ext=".wav"):
    """Transcribe all audio files with specified extension in a directory"""
    audio_files = list(Path(directory).glob(f"*{file_ext}"))
    
    if not audio_files:
        print(f"No {file_ext} files found in {directory}")
        return
    
    results = {}
    
    for audio_file in audio_files:
        print(f"\nProcessing {audio_file.name}...")
        try:
            transcription = transcribe_audio(str(audio_file), model, processor, device)
            results[audio_file.name] = transcription
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            results[audio_file.name] = f"ERROR: {str(e)}"
    
    # Save results to file
    output_file = os.path.join(directory, "transcriptions.txt")
    with open(output_file, "w") as f:
        for filename, transcription in results.items():
            f.write(f"File: {filename}\n")
            f.write(f"Transcription: {transcription}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\nTranscriptions saved to {output_file}")
    return results

def visualize_audio(audio_path, transcription=None):
    """Create a simple visualization of the audio waveform and spectrogram"""
    # Load audio
    audio, sr = load_audio(audio_path)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.imshow(D, aspect='auto', origin='lower', extent=[0, len(audio)/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # Add transcription as subtitle if provided
    if transcription:
        plt.figtext(0.5, 0.01, f"Transcription: {transcription}", 
                   wrap=True, horizontalalignment='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.splitext(audio_path)[0] + "_analysis.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using fine-tuned Whisper model')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to a single audio file')
    group.add_argument('--dir', type=str, help='Path to directory containing audio files')
    
    parser.add_argument('--ext', type=str, default='.wav', help='File extension for batch processing (default: .wav)')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations of audio files')
    parser.add_argument('--compare', type=str, help='Path to ground truth text file for comparison')
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor, device = load_model()
    
    if args.file:
        # Process single file
        transcription = transcribe_audio(args.file, model, processor, device)
        print("\nTranscription result:")
        print("---------------------")
        print(transcription)
        
        # Visualize if requested
        if args.visualize:
            visualize_audio(args.file, transcription)
            
    elif args.dir:
        # Process directory
        results = batch_transcribe(args.dir, model, processor, device, args.ext)
        
        # Visualize if requested
        if args.visualize and results:
            for audio_file in Path(args.dir).glob(f"*{args.ext}"):
                try:
                    visualize_audio(str(audio_file), results.get(audio_file.name))
                except Exception as e:
                    print(f"Error visualizing {audio_file.name}: {e}")


if __name__ == "__main__":
    main()
