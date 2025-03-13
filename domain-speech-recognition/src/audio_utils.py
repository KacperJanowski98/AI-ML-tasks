import os
import numpy as np
import torch
import torchaudio
import librosa
from gtts import gTTS
import io
from pydub import AudioSegment
import matplotlib.pyplot as plt
import soundfile as sf

def text_to_speech(text, output_path=None, sample_rate=16000):
    """
    Convert text to speech using Google Text-to-Speech API
    
    Args:
        text (str): Text to convert to speech
        output_path (str, optional): Path to save the audio file
        sample_rate (int): Target sample rate
    
    Returns:
        numpy.ndarray: Audio signal as a numpy array
        int: Sample rate
    """
    # Generate speech using gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    
    # Save to a bytes buffer
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    # Convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(mp3_fp)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)  # Mono
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / np.max(np.abs(samples))  # Normalize
    
    # Optionally save to file
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, samples, sample_rate)
    
    return samples, sample_rate

def load_and_prepare_noise(noise_path, target_length, sample_rate):
    """
    Load a noise file and prepare it to match the target length
    
    Args:
        noise_path (str): Path to noise audio file
        target_length (int): Target length in samples
        sample_rate (int): Target sample rate
    
    Returns:
        numpy.ndarray: Prepared noise sample
    """
    # Load noise
    if not os.path.exists(noise_path):
        raise FileNotFoundError(f"Noise file not found: {noise_path}")
    
    noise, noise_sr = torchaudio.load(noise_path)
    
    # Convert stereo to mono if needed
    if noise.shape[0] > 1:
        noise = torch.mean(noise, dim=0, keepdim=True)
    
    # Convert to numpy and squeeze
    noise = noise.numpy().squeeze()
    
    # Resample if needed
    if noise_sr != sample_rate:
        noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sample_rate)
    
    # Adjust noise length to match target_length
    if len(noise) > target_length:
        # Randomly select a segment of the noise
        start = np.random.randint(0, len(noise) - target_length)
        noise = noise[start:start + target_length]
    else:
        # Create a new array of the right size and fill it by repeating noise
        result = np.zeros(target_length, dtype=noise.dtype)
        noise_len = len(noise)
        
        # Fill the result array by chunks
        for i in range(0, target_length, noise_len):
            chunk_size = min(noise_len, target_length - i)
            result[i:i+chunk_size] = noise[:chunk_size]
            
        noise = result
    
    return noise

def add_noise_to_speech(speech, noise, snr_db):
    """
    Add noise to speech at a specific signal-to-noise ratio
    
    Args:
        speech (numpy.ndarray): Speech signal
        noise (numpy.ndarray): Noise signal (same length as speech)
        snr_db (float): Target SNR in dB
    
    Returns:
        numpy.ndarray: Noisy speech
    """
    # Calculate speech and noise power
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scale factor
    if noise_power > 0:
        scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
        noisy_speech = speech + scale * noise
    else:
        noisy_speech = speech
    
    # Normalize to avoid clipping
    max_val = np.max(np.abs(noisy_speech))
    if max_val > 1.0:
        noisy_speech = noisy_speech / max_val
    
    return noisy_speech

def plot_waveform(audio, sample_rate, title="Waveform", save_path=None):
    """
    Plot and optionally save a waveform visualization
    
    Args:
        audio (numpy.ndarray): Audio signal
        sample_rate (int): Sample rate
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / sample_rate, len(audio)), audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()