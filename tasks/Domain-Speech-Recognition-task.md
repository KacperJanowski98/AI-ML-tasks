# Domain-Specific Speech Recognition in Challenging Environments

## Overview

Speech recognition technologies have advanced dramatically in recent years, but still face challenges in noisy environments and with specialized vocabulary. This task focuses on fine-tuning a pre-trained Whisper model to improve recognition of domain-specific terminology in challenging acoustic conditions, using techniques like LoRA (Low-Rank Adaptation) or Axolotl.

## Scenario

Law enforcement officers have been equipped with smart wearables (e.g., Apple Watch) to dictate patrol reports in the field. However, the standard speech recognition models struggle with:

1. **Environmental noise**: Wind, traffic, crowd noise during outdoor patrols
2. **Domain-specific terminology**: Police codes, legal terms, procedural language
3. **Hurried/stressed speech**: Officers often need to speak quickly during active situations

Your task is to create a specialized speech recognition pipeline that overcomes these challenges by fine-tuning existing models with a small, targeted dataset.

## Objectives

1. Create a small synthetic dataset of domain-specific audio samples with background noise
2. Fine-tune a pre-trained Whisper model using LoRA or Axolotl to improve domain-specific recognition
3. Implement a pipeline to preprocess noisy audio inputs
4. Evaluate the model on realistic test cases comparing performance with the baseline model

## Technical Requirements

### 1. Dataset Creation

Create a small but effective training dataset (50-100 samples is sufficient for this task):

```python
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor
from datasets import Dataset

def create_synthetic_sample(text, background_noise_path=None, snr_db=10):
    """
    Create a synthetic audio sample with optional background noise
    
    Parameters:
    - text: Text to synthesize
    - background_noise_path: Path to background noise audio file
    - snr_db: Signal-to-noise ratio in decibels
    
    Returns:
    - audio_array: Numpy array of audio with noise
    - sample_rate: Sample rate of the audio
    """
    # Use a TTS system to generate speech from text
    # Here we use a placeholder - you would use an actual TTS system
    # such as gTTS, pyttsx3, or a more advanced model like Facebook's MMS
    speech_array, sample_rate = text_to_speech(text)
    
    if background_noise_path:
        # Load background noise
        noise, noise_sr = torchaudio.load(background_noise_path)
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)
        
        # Adjust noise length to match speech
        if noise.shape[1] > speech_array.shape[0]:
            noise = noise[:, :speech_array.shape[0]]
        else:
            # Loop noise if it's too short
            repeats = int(np.ceil(speech_array.shape[0] / noise.shape[1]))
            noise = noise.repeat(1, repeats)[:, :speech_array.shape[0]]
        
        # Convert to numpy for easier manipulation
        noise = noise.numpy().squeeze()
        
        # Adjust noise level to achieve desired SNR
        speech_power = np.mean(speech_array ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Calculate scaling factor
        scale = np.sqrt(speech_power / (noise_power * (10 ** (snr_db / 10))))
        
        # Mix speech with scaled noise
        noisy_speech = speech_array + scale * noise
        
        return noisy_speech, sample_rate
    
    return speech_array, sample_rate

# Example domain-specific phrases
police_phrases = [
    "Suspect detained at intersection of Main and Broadway, requesting backup.",
    "Code 10-31 in progress at 1420 Elm Street, proceeding with caution.",
    "Officer requesting 10-78, hostile crowd forming near the plaza.",
    "Vehicle matching BOLO description spotted heading eastbound on Highway 42.",
    "Witness describes perpetrator as male, approximately 6'2\", wearing dark hoodie and jeans.",
    # Add more domain-specific phrases
]

# Create a small dataset
dataset_entries = []

for phrase in police_phrases:
    # Generate with different noise types: wind, traffic, crowd, etc.
    noise_files = ["wind_noise.wav", "traffic_noise.wav", "crowd_noise.wav"]
    for noise_file in noise_files:
        for snr in [5, 10, 15]:  # Different noise levels
            audio, sr = create_synthetic_sample(phrase, f"noise_samples/{noise_file}", snr)
            
            # Convert to format expected by Whisper
            processor = AutoProcessor.from_pretrained("openai/whisper-small")
            input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
            
            dataset_entries.append({
                "text": phrase,
                "audio": audio,
                "sampling_rate": sr,
                "input_features": input_features,
                "noise_type": noise_file.split("_")[0],
                "snr": snr
            })

# Convert to HuggingFace dataset
police_dataset = Dataset.from_dict({
    "text": [entry["text"] for entry in dataset_entries],
    "audio": [entry["audio"] for entry in dataset_entries],
    "sampling_rate": [entry["sampling_rate"] for entry in dataset_entries],
    "noise_type": [entry["noise_type"] for entry in dataset_entries],
    "snr": [entry["snr"] for entry in dataset_entries],
})
```

Alternative dataset approach: Record actual domain-specific phrases and mix with real-world noise samples or use existing datasets like DEMAND or WHAM! for noise augmentation.

### 2. Fine-tuning with LoRA

LoRA is ideal for this task as it allows efficient fine-tuning of large models with fewer parameters:

```python
from peft import get_peft_model, LoraConfig, TaskType

def prepare_lora_model(base_model_id="openai/whisper-small"):
    """Prepare a Whisper model with LoRA configuration"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    # Load pre-trained model
    model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    
    # Define LoRA Config
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention query and value matrices
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Wrap model with LoRA
    lora_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print(f"Trainable parameters: {lora_model.print_trainable_parameters()}")
    
    return lora_model, WhisperProcessor.from_pretrained(base_model_id)
```

### 3. Fine-tuning with Axolotl (Alternative)

For those who prefer using Axolotl, here's an example configuration:

```yaml
# axolotl_config.yml
base_model: openai/whisper-small
model_type: whisper
tokenizer: openai/whisper-small

load_in_8bit: false
adapter: lora

lora_model_dir: ./lora-whisper
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj

sequence_len: 256
sample_packing: false
pad_to_sequence_len: true

datasets:
  - path: police_reports_dataset
    type: json

dataset_prepared_path: ./prepared_dataset
val_set_size: 0.05

output_dir: ./whisper-police-lora
training_args:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 1
  warmup_steps: 10
  max_steps: 200
  learning_rate: 3.0e-5
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 50
  save_strategy: steps
  save_steps: 50
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: wer
  greater_is_better: false
```

And the command to run it:

```bash
axolotl train axolotl_config.yml
```

### 4. Apple Silicon / MLX-LM Implementation (Optional)

For those using Apple Silicon hardware, leverage the MLX-LM framework for efficient training:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model

# Load whisper model
model, tokenizer = load_model("openai/whisper-small")

# Define LoRA layers for MLX
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=32):
        super().__init__()
        self.linear = linear
        in_features, out_features = linear.weight.shape
        scale = alpha / rank
        
        # LoRA components
        self.lora_A = mx.random.normal(shape=(in_features, rank)) * 0.01
        self.lora_B = mx.zeros(shape=(rank, out_features))
        self.scale = scale
        
    def __call__(self, x):
        # Original linear operation
        original = self.linear(x)
        
        # LoRA path: x @ A @ B
        lora = x @ self.lora_A @ self.lora_B * self.scale
        
        # Combine
        return original + lora

# Apply LoRA to attention modules
def apply_lora_to_whisper(model, rank=4, alpha=32):
    for layer in model.encoder.blocks:
        # Apply to attention layers
        if hasattr(layer, "attn"):
            if hasattr(layer.attn, "query"):
                layer.attn.query = LoRALinear(layer.attn.query, rank, alpha)
            if hasattr(layer.attn, "value"):
                layer.attn.value = LoRALinear(layer.attn.value, rank, alpha)
    return model

# Apply LoRA
model = apply_lora_to_whisper(model)

# Training function specific to MLX
def train_step(model, inputs, targets, optimizer):
    def loss_fn(model):
        logits = model(inputs)
        return nn.losses.cross_entropy(logits, targets)
    
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Main training loop will be implemented here
```

### 5. Audio Preprocessing Pipeline

Create a robust preprocessing pipeline for handling noisy inputs:

```python
def preprocess_audio(audio_path, target_sr=16000):
    """
    Process audio for better speech recognition performance in noisy environments
    
    Steps:
    1. Resample to target sample rate
    2. Apply noise reduction
    3. Apply normalization
    4. Optional spectral subtraction for background noise
    """
    import librosa
    import noisereduce as nr
    
    # Load audio and resample
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Apply noise reduction (noisereduce library uses spectral gating)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    
    # Normalize audio
    normalized = librosa.util.normalize(reduced_noise)
    
    # Optional: Detect and enhance speech segments
    # This example uses a simple energy-based VAD
    # For better results, you might use a pre-trained VAD model
    energy = librosa.feature.rms(y=normalized)[0]
    speech_segments = energy > 0.01  # Simple threshold-based detection
    
    # Apply slight enhancement to speech segments
    enhanced = normalized.copy()
    for i in range(len(speech_segments)):
        if speech_segments[i]:
            frame_start = i * 512  # Assuming frame length of 512
            frame_end = min(frame_start + 512, len(enhanced))
            enhanced[frame_start:frame_end] *= 1.2  # Slight boost to speech
    
    # Ensure we don't clip
    enhanced = librosa.util.normalize(enhanced)
    
    return enhanced, target_sr
```

### 6. Evaluation Framework

Create a comprehensive evaluation framework:

```python
def evaluate_model(model, processor, test_files, ground_truth):
    """
    Evaluate speech recognition model on test files
    
    Parameters:
    - model: The fine-tuned Whisper model
    - processor: Whisper processor
    - test_files: List of audio file paths
    - ground_truth: List of corresponding transcripts
    
    Returns:
    - Dictionary of metrics including WER, domain-specific term accuracy, etc.
    """
    from jiwer import wer
    import re
    
    results = {
        "wer": [],
        "domain_term_accuracy": [],
        "predictions": [],
        "ground_truth": []
    }
    
    # Define domain-specific terms to track
    domain_terms = [
        "code 10-31", "backup", "suspect", "officer", "bolo", 
        "perpetrator", "witness", "patrol", "dispatch", "precinct"
    ]
    
    for audio_file, reference in zip(test_files, ground_truth):
        # Preprocess audio
        audio, sr = preprocess_audio(audio_file)
        
        # Prepare for model
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        
        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode prediction
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Calculate WER
        error = wer(reference, prediction)
        results["wer"].append(error)
        
        # Calculate domain term accuracy
        ref_terms = sum([1 for term in domain_terms if term.lower() in reference.lower()])
        correct_terms = 0
        
        if ref_terms > 0:
            for term in domain_terms:
                if term.lower() in reference.lower() and term.lower() in prediction.lower():
                    correct_terms += 1
            
            term_accuracy = correct_terms / ref_terms if ref_terms > 0 else 0
            results["domain_term_accuracy"].append(term_accuracy)
        
        # Store predictions for further analysis
        results["predictions"].append(prediction)
        results["ground_truth"].append(reference)
    
    # Calculate averages
    results["avg_wer"] = sum(results["wer"]) / len(results["wer"])
    if results["domain_term_accuracy"]:
        results["avg_domain_accuracy"] = sum(results["domain_term_accuracy"]) / len(results["domain_term_accuracy"])
    else:
        results["avg_domain_accuracy"] = 0
    
    return results
```

## Expected Deliverables

1. **Code Repository**:
   - Scripts for dataset creation/collection
   - Fine-tuning implementation (LoRA or Axolotl)
   - Audio preprocessing pipeline
   - Evaluation framework

2. **Dataset**:
   - Small corpus of domain-specific phrases (text)
   - Generated or collected audio samples with different noise conditions
   - Test set for evaluation

3. **Trained Model**:
   - Fine-tuned Whisper model (or LoRA adapters)
   - Sample inference code

4. **Evaluation Report**:
   - Comparison of baseline vs fine-tuned model
   - Analysis of performance across different noise conditions
   - Domain-specific term recognition accuracy
   - Recommendations for further improvements

## Evaluation Criteria

Your solution will be evaluated based on:

1. **Improvement over Baseline**: How much did your fine-tuning improve domain-specific recognition in noisy conditions?
2. **Technical Implementation**: Quality of implementation, including dataset preparation, preprocessing, and fine-tuning approach
3. **Problem Solving**: Creative solutions to handle challenging audio conditions
4. **Model Efficiency**: Resource requirements for training and inference
5. **Documentation Quality**: Clear explanation of approach, challenges, and results

## Time Expectation

- This task should take approximately 4-6 hours for an experienced ML engineer
- Focus areas if time is limited:
  - Dataset preparation with a few good examples
  - Basic LoRA implementation
  - Simple but effective evaluation

## Notes and Hints

- You don't need a large dataset for effective domain adaptation with LoRA
- Consider the balance between noise augmentation (too little won't generalize, too much might be unrealistic)
- The quality of your test set design is as important as the model training process
- If using Apple Silicon, the MLX framework can provide significant speedups and memory efficiency

## Bonus Challenges

If you complete the main task and want to push further:

1. Implement a real-time streaming inference pipeline
2. Create a simple UI for testing the model (e.g., upload audio or record directly)
3. Develop a confidence scoring system to flag potential transcription errors for human review
4. Compare performance across multiple base models (tiny, small, medium Whisper variants)
