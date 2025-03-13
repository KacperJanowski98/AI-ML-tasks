import os
import torch
import numpy as np
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Union

# Configuration
BASE_MODEL = "openai/whisper-small"
DATASET_DIR = "data/processed_dataset"
OUTPUT_DIR = "models/whisper-lora-simple"

@dataclass
class DebugDataCollatorWhisperPadding:
    processor: any
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Debug information
        print("Processing batch...")
        
        # First, process the audio
        input_arrays = []
        input_sampling_rates = []
        for feature in features:
            audio = feature["audio"]
            input_arrays.append(audio["array"])
            input_sampling_rates.append(audio["sampling_rate"])
        
        # Extract transcriptions
        label_features = [feature["text"] for feature in features]
        
        # Debug: print audio length info
        for i, arr in enumerate(input_arrays):
            print(f"Audio {i}: Length={len(arr)}, SR={input_sampling_rates[i]}")
        
        # Process features using feature extractor 
        # THIS IS THE CRITICAL PART - we need to make sure we get 3000 frames
        batch_features = self.processor.feature_extractor(
            input_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Debug: print feature shape
        print(f"Original feature shape: {batch_features.input_features.shape}")
        
        # Manually ensure we have exactly 3000 frames for each audio
        # The shape should be [batch_size, num_mel_bins, time_frames]
        # So we're padding/truncating the last dimension to 3000
        batch_size, num_mel_bins, time_frames = batch_features.input_features.shape
        
        # Create properly sized tensor
        padded_features = torch.zeros((batch_size, num_mel_bins, 3000), dtype=batch_features.input_features.dtype)
        
        # Copy data, handling both truncation and padding
        for i in range(batch_size):
            # If shorter than 3000, copy and pad
            if time_frames < 3000:
                padded_features[i, :, :time_frames] = batch_features.input_features[i]
            # If longer than 3000, truncate
            else:
                padded_features[i, :, :3000] = batch_features.input_features[i, :, :3000]
        
        # Debug: print new feature shape
        print(f"Padded feature shape: {padded_features.shape}")
        
        # Process labels
        label_tokens = self.processor.tokenizer(
            text=label_features,
            padding=True,
            return_tensors="pt"
        )
        
        # Replace padding token with -100 for loss calculation
        labels = label_tokens.input_ids.masked_fill(
            label_tokens.input_ids == self.processor.tokenizer.pad_token_id,
            -100
        )
        
        # Create batch
        batch = {
            "input_features": padded_features,
            "labels": labels
        }
        
        return batch

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_DIR)
    
    # Load processor and model
    print("Loading Whisper model and processor...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    
    # Prepare audio column
    print("Preparing audio data...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Set up LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    # Add LoRA to model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create data collator with debugging
    data_collator = DebugDataCollatorWhisperPadding(processor=processor)
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # Smaller batch size for testing
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=50,
        num_train_epochs=1,  # Just 1 epoch for testing
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        generation_max_length=225,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=True,
        remove_unused_columns=False,
    )
    
    # Set up WER metric
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with padding token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
