# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fine-tuning laboratory project for training TinyLlama-1.1B-Chat-v1.0 to write Chinese five-character poems (五言诗) using LoRA (Low-Rank Adaptation). The project is designed to run entirely on CPU without requiring GPU resources.

## Development Setup

### Environment Setup with uv
```bash
# Initialize project (already done)
uv init --name fine-tuning-lab --python 3.11

# Install dependencies (already done)
uv add transformers peft accelerate torch wandb datasets

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Or use: uv run python script.py
```

### Weights & Biases Setup
```bash
wandb login
```
You'll need a free W&B account to track training metrics.

## Common Commands

### Run Complete Fine-tuning Pipeline
```bash
# Using uv (recommended)
uv run python tinyllama_poetry_finetune.py

# Or activate venv first
.venv\Scripts\activate
python tinyllama_poetry_finetune.py
```
This will:
1. Test the model before fine-tuning
2. Fine-tune using LoRA on 5 Chinese poetry examples
3. Test the model after fine-tuning
4. Save the adapter to `./tinyllama-poetry/`
5. Log metrics to W&B

### Test Fine-tuned Model Interactively
```bash
# Using uv (recommended)
uv run python test_model.py

# Or with activated venv
python test_model.py
```

### View Training Metrics
Check your W&B dashboard at: https://wandb.ai/

## Architecture

### Model Details
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Chinese five-character poetry generation
- **Training Device**: CPU only
- **Dataset**: 5 hand-crafted Chinese poem examples

### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Dropout: 0.1
- Target modules: All attention layers (q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj)

### Training Parameters
- Epochs: 10
- Batch size: 1 (with 4 gradient accumulation steps)
- Learning rate: 2e-4
- Scheduler: Cosine
- Mixed precision: Disabled (CPU training)

## File Structure

- `tinyllama_poetry_finetune.py` - Main fine-tuning script
- `test_model.py` - Interactive testing script
- `training_data.json` - Training dataset with Chinese poetry examples
- `./tinyllama-poetry/` - Output directory for fine-tuned adapter
- `.venv/` - Python virtual environment (created by uv)
- `pyproject.toml` - Project configuration and dependencies
- `CLAUDE.md` - This documentation file

## Training Data Management

### Modifying Training Data
Edit `training_data.json` to:
- Add new poetry examples
- Change existing poems
- Add different topics or themes

### JSON Structure
```json
{
  "examples": [
    {
      "id": "unique_identifier",
      "topic": "Topic (English/Chinese)",
      "input": "Write a Chinese five-character poem about [topic]",
      "output": "五言诗内容，\n每行五个字，\n通常四行诗，\n传统诗歌格式。"
    }
  ]
}
```

## Important Notes

- **CPU Only**: All training is configured for CPU-only execution
- **LoRA Adapters**: Only the adapter weights are saved, not the full model
- **Chinese Poetry Format**: Training focuses on 五言诗 (5-character lines, typically 4 lines per poem)
- **Chat Format**: Uses TinyLlama's specific chat template for consistency
- **Small Dataset**: Uses only 5 examples for demonstration; expand for better results
- **W&B Logging**: All training metrics are logged to Weights & Biases for monitoring