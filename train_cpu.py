import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# Configuration
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DATASET_PATH = "mind2web_execution_training.jsonl"
OUTPUT_DIR = "./qwen2-0.5b-cpu-test"

def train():
    print(f"Loading tiny model {MODEL_ID} for CPU training...")
    # 1. Load Dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Model (No quantization, CPU only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True, # Added this
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=1,
        max_steps=2,
        save_steps=10,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        # Added these for memory saving
        gradient_checkpointing=True,
    )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['messages'])):
            messages = example['messages'][i]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(formatted)
        return output_texts

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=256, # Very short for CPU test
        tokenizer=tokenizer,
        args=training_args,
    )

    # 6. Start Training
    print("Starting CPU training test (2 steps)...")
    trainer.train()

    # 7. Save Model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train()
