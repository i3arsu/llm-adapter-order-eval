#!/usr/bin/env python3
"""
Specialized Deepseek retraining script with strict JSON output constraints.
This script retrains the Deepseek adapter with:
1. Explicit JSON format enforcement in system prompt
2. Examples showing correct JSON output
3. NO thinking tags in response
"""

import torch
import pandas as pd
import os
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_TYPE = "lora"
MODEL_SHORT_NAME = "deepseek"
OUTPUT_DIR = f"./{ADAPTER_TYPE}-{MODEL_SHORT_NAME}-checkpoints-json"

print("=" * 70)
print("DEEPSEEK JSON RETRAINING")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# Login
login(token=os.getenv("HF_TOKEN"))

# 1. Data Preparation with Strict JSON Formatting
# ---------------------------------------------------------
print("Loading and cleaning data...")
df = pd.read_csv("Retail_Dataset_10000.csv")

def is_data_valid(row):
    return str(row['product']).lower() in str(row['user_input']).lower()

df = df[df.apply(is_data_valid, axis=1)]
print(f"Total valid samples: {len(df)}")

raw_dataset = Dataset.from_pandas(df[['user_input', 'action', 'product', 'quantity']])
dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
raw_train_dataset = dataset_split['train']
raw_eval_dataset = dataset_split['test']

print(f"Training set: {len(raw_train_dataset)} | Validation: {len(raw_eval_dataset)}\n")

# 2. Model Setup
# ---------------------------------------------------------
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# STRICT SYSTEM PROMPT - Forces JSON-only output with explicit constraints
SYSTEM_PROMPT = """You are a JSON extraction assistant. RESPOND ONLY WITH VALID JSON. NO TEXT BEFORE OR AFTER.

STRICT RULES (MANDATORY):
1. Output ONLY a JSON object with no other text whatsoever
2. NEVER include explanations, thinking, or any text outside JSON
3. NEVER include <think>, <answer>, or any tags
4. NEVER use unquoted keys - ALL keys must be in double quotes
5. NEVER use unquoted string values - ALL strings must be in double quotes
6. JSON fields MUST be: "action", "product", "quantity"
7. "action" value MUST be either "add" or "remove" (lowercase, quoted)
8. "product" value MUST be a quoted string (e.g., "milk", "bread")
9. "quantity" value MUST be a number without quotes (e.g., 5, not "5")
10. Use proper JSON syntax with colons, commas, curly braces

VALID EXAMPLES:
{"action": "add", "product": "milk", "quantity": 5}
{"action": "remove", "product": "bread", "quantity": 2}
{"action": "add", "product": "juice", "quantity": 20}

INVALID EXAMPLES (DO NOT OUTPUT THESE):
{action: "add", product: "milk", quantity: 5}  <- Unquoted keys are WRONG
{"action": "add", "product": "milk", "quantity": "5"}  <- Quoted number is WRONG
<think>...</think>{"action": "add"...}  <- Thinking tags are WRONG
I need to add 5 milk {"action": "add"...}  <- Text before JSON is WRONG"""

def format_with_strict_json(example):
    """Format training data with strict JSON output constraints and few-shot examples."""
    
    # Few-shot examples to teach the model correct JSON format
    few_shot_examples = """
EXAMPLE 1:
User: Add 5 milk
Response: {"action": "add", "product": "milk", "quantity": 5}

EXAMPLE 2:
User: Remove 3 bread
Response: {"action": "remove", "product": "bread", "quantity": 3}

EXAMPLE 3:
User: I want 10 apples
Response: {"action": "add", "product": "apples", "quantity": 10}
"""
    
    enhanced_prompt = SYSTEM_PROMPT + "\n" + few_shot_examples
    
    messages = [
        {"role": "system", "content": enhanced_prompt},
        {"role": "user", "content": example['user_input']},
        # Response MUST be valid JSON only - precisely formatted
        {"role": "assistant", "content": f'{{"action": "{example["action"]}", "product": "{example["product"]}", "quantity": {example["quantity"]}}}'}
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        # Fallback - simple format
        print(f"Warning: Chat template failed, using simple format. Error: {e}")
        text = f"{enhanced_prompt}\n\nUser: {example['user_input']}\n\nResponse: {{\"action\": \"{example['action']}\", \"product\": \"{example['product']}\", \"quantity\": {example['quantity']}}}"
    return {"text": text}

print("Formatting training data with strict JSON constraints...")
train_dataset = raw_train_dataset.map(format_with_strict_json, remove_columns=raw_train_dataset.column_names)
eval_dataset = raw_eval_dataset.map(format_with_strict_json, remove_columns=raw_eval_dataset.column_names)

print("Loading model in 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA Configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Training Configuration
# ---------------------------------------------------------
print("\nConfiguring training parameters...")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # --- DATASET PARAMETERS ---
    dataset_text_field="text",
    max_length=512,
    packing=False,
    # --------------------------

    # --- BATCH SETTINGS (RTX 3070 compatible) ---
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 2*8 = 16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # ------------------------------------------

    # --- TRAINING PARAMETERS ---
    num_train_epochs=3,
    learning_rate=5e-4,
    bf16=True,
    logging_steps=25,
    optim="paged_adamw_32bit",
    
    # --- VALIDATION ---
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    report_to="none"
)

# 4. Train
# ---------------------------------------------------------
print("Starting training with strict JSON-only output constraints...")
print(f"- System prompt with explicit rules against unquoted keys")
print(f"- Few-shot examples included in training data")
print(f"- 3 epochs with focus on JSON format enforcement\n")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config
)

trainer.train()

# 5. Save
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("Training complete! Saving model...")
print("=" * 70)

os.makedirs("./adapters", exist_ok=True)
new_model_name = f"./adapters/{ADAPTER_TYPE}-{MODEL_SHORT_NAME}-merged-json"

# Save the adapter
trainer.model.save_pretrained(f"{new_model_name}")
tokenizer.save_pretrained(f"{new_model_name}")

print(f"\n✓ Model saved to: {new_model_name}")
print(f"\nIMPORTANT NOTES:")
print(f"1. This model was trained with strict JSON-only output constraints")
print(f"2. No thinking tags should appear in the output")
print(f"3. All outputs should be valid JSON format")
print(f"4. Test with: python3 debug_deepseek.py")
