import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./adapters/llama3-retail-adapter_a100"

# 1. Data Preparation
# ---------------------------------------------------------
print("Loading and cleaning data...")
df = pd.read_csv("Retail_Dataset_10000.csv")

def is_data_valid(row):
    return str(row['product']).lower() in str(row['user_input']).lower()

df = df[df.apply(is_data_valid, axis=1)]

def format_instruction(row):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Analyze the user request and extract action, product, and quantity into JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>

{row['user_input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    "action": "{row['action']}",
    "product": "{row['product']}",
    "quantity": {row['quantity']}
}}<|eot_id|>"""

df['text'] = df.apply(format_instruction, axis=1)
full_dataset = Dataset.from_pandas(df[['text']])

dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

print(f"Training set: {len(train_dataset)} | Validation: {len(eval_dataset)}")

# 2. Model and Config - OPTIMIZED FOR A100
# ---------------------------------------------------------
# Option A: Use BF16 precision (no quantization) for maximum speed
# This will use ~16GB VRAM, leaving plenty of room for larger batches
print("Loading model in BF16 precision (optimized for A100)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # Native BF16 support on A100
    device_map="auto",
    use_cache=False,
    attn_implementation="flash_attention_2"  # A100 supports Flash Attention 2
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# LoRA Configuration - optimized for better performance
peft_config = LoraConfig(
    r=32,  # Increased rank for better capacity
    lora_alpha=64,  # Increased alpha proportionally
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Training Configuration - A100 OPTIMIZED
# ---------------------------------------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # --- DATASET PARAMETERS ---
    dataset_text_field="text",
    max_seq_length=1024,  # Increased from 512 (A100 can handle this easily)
    packing=True,  # Enable packing for efficiency with variable-length sequences
    # --------------------------

    # --- A100 OPTIMIZED BATCH SETTINGS ---
    per_device_train_batch_size=8,  # Increased from 1 (A100 has 40GB memory)
    per_device_eval_batch_size=8,   # Match training batch size
    gradient_accumulation_steps=4,  # Reduced from 16 (effective batch = 8*4 = 32)
    gradient_checkpointing=True,
    # -------------------------------------

    # --- TRAINING PARAMETERS ---
    num_train_epochs=3,  # Increased for better convergence
    learning_rate=2e-4,
    warmup_ratio=0.03,  # Add warmup for stability
    fp16=False,
    bf16=True,  # A100 has excellent BF16 performance
    tf32=True,  # Enable TF32 for additional speedup on A100
    
    # --- OPTIMIZATION ---
    optim="adamw_torch_fused",  # Faster than paged_adamw on A100
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # -------------------
    
    # --- LOGGING AND EVALUATION ---
    logging_steps=10,  # More frequent logging
    eval_strategy="steps", 
    eval_steps=25,  # More frequent evaluation
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,  # Keep only best 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # -----------------------------
    
    # --- PERFORMANCE ---
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster data transfer to GPU
    group_by_length=True,  # Group similar lengths for efficiency
    # ------------------
    
    report_to="tensorboard",  # Enable TensorBoard logging
    logging_dir=f"{OUTPUT_DIR}/logs"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config
)

print("Starting A100-optimized training...")
print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"Total training steps: ~{len(train_dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps) * sft_config.num_train_epochs}")

trainer.train()

# 4. Save Final Model
# ---------------------------------------------------------
new_model_name = "llama3-retail-a100-final"
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"Training complete! Model saved to: {new_model_name}")

# Optional: Save merged model (LoRA + base model)
print("\nMerging LoRA adapters with base model...")
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(f"{new_model_name}-merged")
tokenizer.save_pretrained(f"{new_model_name}-merged")
print(f"Merged model saved to: {new_model_name}-merged")