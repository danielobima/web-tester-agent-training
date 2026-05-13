# Qwen Fine-Tuning for AI-QA Agent

This directory contains the scripts and architecture required to fine-tune a Qwen2 model on the Mind2Web dataset, tailored for the AI-QA execution agent.

## Architecture Overview

1.  **Data Processing**: `format_mind2web.py` converts the raw dataset into a conversational format (`messages`) that includes:
    - System prompt (populated from `ai-qa-bot/src/prompts/execution.txt`).
    - Multi-turn history.
    - Exact text blocks (Goal/Task/State) used at runtime.
    - Ground-truth JSON actions.

2.  **Training Stack**:
    - **Model**: Qwen2-7B-Instruct (or other Qwen2 variants).
    - **Optimization**: 4-bit Quantization (QLoRA) to allow training on consumer GPUs (e.g., 24GB VRAM).
    - **Trainer**: `trl.SFTTrainer` (Supervised Fine-Tuning) with PEFT (Parameter-Efficient Fine-Tuning).
    - **Format**: Standard ChatML format supported by Qwen.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_train.txt
```

### 2. Prepare Data
Run the formatter to generate the training file:
```bash
python3 format_mind2web.py
```
*Note: You can modify `format_mind2web.py` to process the full dataset by changing `split="train[:2]"` to `split="train"`.*

### 3. Run Training
```bash
python3 train_qwen.py
```
The script will save LoRA adapters to `./qwen2-7b-mind2web`.

### 4. Test Inference
```bash
python3 test_qwen.py
```

## Key Configuration
- **Max Sequence Length**: 4096 (to accommodate large DOM snapshots).
- **Quantization**: NF4 4-bit.
- **LoRA Rank (r)**: 16 (adjustable).
- **Target Modules**: All linear layers in the transformer block are targeted for better performance.
