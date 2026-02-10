import torch
import pandas as pd
from datasets import Dataset
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./adapters/llama3-retail-ia3-a100"

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

# 2. Model and IA3 Config - OPTIMIZED FOR A100
# ---------------------------------------------------------
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

# IA3 Configuration
# IA3 works by learning scaling vectors for key, value, and feedforward layers
ia3_config = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "v_proj", "down_proj"],  # Standard IA3 targets
    feedforward_modules=["down_proj"],  # FFN modules to adapt
    inference_mode=False,
)

print("Applying IA3 adapter...")
model = get_peft_model(model, ia3_config)

# Print trainable parameters
model.print_trainable_parameters()

# 3. Training Configuration - A100 OPTIMIZED FOR IA3
# ---------------------------------------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # --- DATASET PARAMETERS ---
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True,
    # --------------------------

    # --- A100 OPTIMIZED BATCH SETTINGS ---
    # IA3 has fewer parameters, so we can use larger batches
    per_device_train_batch_size=16,  # Doubled from LoRA version
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,   # Effective batch = 16*2 = 32
    gradient_checkpointing=True,
    # -------------------------------------

    # --- TRAINING PARAMETERS ---
    num_train_epochs=3,
    learning_rate=8e-3,  # IA3 typically uses higher LR than LoRA (1e-4 to 1e-2)
    warmup_ratio=0.03,
    fp16=False,
    bf16=True,
    tf32=True,
    
    # --- OPTIMIZATION ---
    optim="adamw_torch_fused",
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.0,  # IA3 often works better without weight decay
    max_grad_norm=1.0,
    # -------------------
    
    # --- LOGGING AND EVALUATION ---
    logging_steps=10,
    eval_strategy="steps", 
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # -----------------------------
    
    # --- PERFORMANCE ---
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=True,
    # ------------------
    
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/logs"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=sft_config
)

print("Starting A100-optimized IA3 training...")
print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"Total training steps: ~{len(train_dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps) * sft_config.num_train_epochs}")

trainer.train()

# 4. Save Final Model
# ---------------------------------------------------------
new_model_name = "llama3-retail-ia3-a100-final"
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"Training complete! IA3 adapter saved to: {new_model_name}")

# Print final adapter size
import os
adapter_size = os.path.getsize(f"{new_model_name}/adapter_model.safetensors") / (1024*1024)
print(f"\nIA3 Adapter size: {adapter_size:.2f} MB")

# Optional: Save merged model
print("\nMerging IA3 adapter with base model...")
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(f"{new_model_name}-merged")
tokenizer.save_pretrained(f"{new_model_name}-merged")
print(f"Merged model saved to: {new_model_name}-merged")