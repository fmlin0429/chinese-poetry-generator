"""
TinyLlama Chinese Five-Character Poetry Fine-tuning Script
=========================================================

This script fine-tunes the TinyLlama-1.1B-Chat-v1.0 model to write Chinese five-character poems (五言诗)
using LoRA (Low-Rank Adaptation) for efficient training on CPU only.

Requirements:
- transformers, peft, accelerate, torch, wandb, datasets (pip install)
- CPU-only training (no GPU required)
- Weights & Biases account for logging

Usage:
1. Run this script step by step in Cursor
2. First section tests the model before training
3. Second section performs LoRA fine-tuning
4. Third section tests the improved model
"""

import torch
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os
import sys
from datetime import datetime

# Fix Windows Unicode encoding issues
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-poetry"
WANDB_PROJECT = "tinyllama-chinese-poetry"
TRAINING_DATA_FILE = "./training_data.json"

# Force CPU usage (no GPU)
device = torch.device("cpu")
print(f"Using device: {device}")

# =============================================================================
# Step 1: Load Model and Tokenizer
# =============================================================================

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map=None,            # Don't use device mapping for CPU
    trust_remote_code=True
)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded with {model.num_parameters():,} parameters")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# =============================================================================
# Step 2: Test Model BEFORE Fine-tuning
# =============================================================================

def generate_poem(model, tokenizer, prompt, max_length=100):
    """Generate text using the model"""
    # Format prompt in chat format that TinyLlama expects
    chat_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    inputs = tokenizer.encode(chat_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            early_stopping=True
        )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

print("\n" + "="*60)
print("TESTING MODEL BEFORE FINE-TUNING")
print("="*60)

test_prompts = [
    "Write a Chinese five-character poem about spring (春天)",
    "Write a poem about flowers (花)",
    "Create a five-character poem about mountains (山)"
]

print("Model responses BEFORE fine-tuning:")
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Prompt: {prompt}")
    response = generate_poem(model, tokenizer, prompt)
    print(f"   Response: {response}")

# =============================================================================
# Step 3: Prepare Training Data
# =============================================================================

print("\n" + "="*60)
print("PREPARING TRAINING DATA")
print("="*60)

# Load training data from external JSON file
print(f"Loading training data from: {TRAINING_DATA_FILE}")
try:
    with open(TRAINING_DATA_FILE, 'r', encoding='utf-8') as f:
        training_data_json = json.load(f)
    
    # Extract training examples from JSON structure
    training_data = []
    for example in training_data_json["examples"]:
        training_data.append({
            "input": example["input"],
            "output": example["output"]
        })
    
    print(f"Successfully loaded {len(training_data)} training examples")
    print("Topics included:", [ex.get("topic", "Unknown") for ex in training_data_json["examples"]])
    
except FileNotFoundError:
    print(f"Error: Training data file '{TRAINING_DATA_FILE}' not found!")
    print("Please make sure the training_data.json file exists.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in training data file: {e}")
    exit(1)
except KeyError as e:
    print(f"Error: Missing required field in training data: {e}")
    exit(1)

def format_training_example(example):
    """Format training example in chat format"""
    prompt = example["input"]
    response = example["output"]
    
    # Use TinyLlama's chat format
    formatted = f"<|system|>\nYou are a helpful assistant that writes Chinese five-character poems.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n{response}</s>"
    return {"text": formatted}

# Convert to dataset
formatted_data = [format_training_example(example) for example in training_data]
train_dataset = Dataset.from_list(formatted_data)

print(f"Created training dataset with {len(train_dataset)} examples")
print("Sample training example:")
print(train_dataset[0]["text"])

# =============================================================================
# Step 4: Tokenize Dataset
# =============================================================================

def tokenize_function(examples):
    """Tokenize the training examples"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # We'll pad in the data collator
        max_length=256,
        return_tensors=None  # Return lists instead of tensors for batching
    )
    # For language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("\nTokenizing dataset...")
tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

print(f"Tokenized dataset: {tokenized_dataset}")

# =============================================================================
# Step 5: Configure LoRA
# =============================================================================

print("\n" + "="*60)
print("CONFIGURING LORA")
print("="*60)

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal language modeling
    inference_mode=False,          # Training mode
    r=16,                         # LoRA rank (higher = more parameters but better performance)
    lora_alpha=32,                # LoRA scaling parameter
    lora_dropout=0.1,             # LoRA dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # TinyLlama attention modules
    bias="none",                  # No bias adaptation
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# =============================================================================
# Step 6: Initialize Weights & Biases
# =============================================================================

print("\n" + "="*60)
print("INITIALIZING WEIGHTS & BIASES")
print("="*60)

# Initialize W&B (you'll need to log in first with: wandb login)
wandb.init(
    project=WANDB_PROJECT,
    name=f"tinyllama-poetry-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config={
        "model_name": MODEL_NAME,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": lora_config.target_modules,
        "dataset_size": len(train_dataset),
        "device": str(device),
        "task": "Chinese five-character poetry generation"
    }
)

# =============================================================================
# Step 7: Configure Training Arguments
# =============================================================================

print("Configuring training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,           # More epochs due to small dataset
    per_device_train_batch_size=1, # Small batch size for CPU
    gradient_accumulation_steps=4,  # Simulate larger batch size
    warmup_steps=10,
    logging_steps=1,               # Log every step due to small dataset
    save_steps=50,
    eval_steps=50,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,   # Disable for CPU
    dataloader_num_workers=0,      # Single-threaded for CPU
    report_to="wandb",             # Log to W&B
    run_name=f"tinyllama-poetry-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    learning_rate=2e-4,            # Standard LoRA learning rate
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=False,                    # Disable mixed precision for CPU
    push_to_hub=False,
    load_best_model_at_end=False,
)

# =============================================================================
# Step 8: Set Up Data Collator and Trainer
# =============================================================================

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're not doing masked language modeling
    pad_to_multiple_of=None,
    return_tensors="pt"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# =============================================================================
# Step 9: Start Fine-tuning
# =============================================================================

print("\n" + "="*60)
print("STARTING FINE-TUNING")
print("="*60)

print("Starting training... (This will take some time on CPU)")
print("Monitor progress on Weights & Biases dashboard")

# Start training
trainer.train()

print("Training completed!")

# =============================================================================
# Step 10: Save the Fine-tuned Model
# =============================================================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save the LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")

# =============================================================================
# Step 11: Test Model AFTER Fine-tuning
# =============================================================================

print("\n" + "="*60)
print("TESTING MODEL AFTER FINE-TUNING")
print("="*60)

print("Model responses AFTER fine-tuning:")
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Prompt: {prompt}")
    response = generate_poem(model, tokenizer, prompt)
    print(f"   Response: {response}")

# =============================================================================
# Step 12: Compare Results
# =============================================================================

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

print("Fine-tuning completed successfully!")
print(f"- Model: {MODEL_NAME}")
print(f"- Training examples: {len(training_data)}")
print(f"- Output directory: {OUTPUT_DIR}")
print(f"- W&B project: {WANDB_PROJECT}")
print(f"- Device used: {device}")

print("\nNext steps:")
print("1. Check the W&B dashboard for training metrics")
print("2. Test the model with different prompts")
print("3. The LoRA adapter is saved locally and can be reused")

# Finish W&B run
wandb.finish()

print("\nScript completed! Check the outputs above to compare before/after results.")