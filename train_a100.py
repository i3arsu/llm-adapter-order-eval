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
import time
import os
import json

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

# Print trainable parameters before training
print("\n" + "="*70)
trainer.model.print_trainable_parameters()
print("="*70 + "\n")

# Start timing
training_start_time = time.time()

# Train
trainer.train()

# Calculate training time
training_end_time = time.time()
total_training_time = training_end_time - training_start_time

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

# 5. Calculate and Display Metrics
# ---------------------------------------------------------
print("\n" + "="*70)
print("TRAINING METRICS SUMMARY")
print("="*70)

# Training time
hours = int(total_training_time // 3600)
minutes = int((total_training_time % 3600) // 60)
seconds = int(total_training_time % 60)
print(f"\n📊 Training Time:")
print(f"   Total: {total_training_time:.2f} seconds ({hours}h {minutes}m {seconds}s)")
print(f"   Per Epoch: {total_training_time / sft_config.num_train_epochs:.2f} seconds")

# Adapter size
adapter_path = f"{new_model_name}/adapter_model.safetensors"
if os.path.exists(adapter_path):
    adapter_size_bytes = os.path.getsize(adapter_path)
    adapter_size_mb = adapter_size_bytes / (1024 * 1024)
    print(f"\n💾 Adapter Size:")
    print(f"   {adapter_size_mb:.2f} MB ({adapter_size_bytes:,} bytes)")
else:
    print(f"\n⚠️  Adapter file not found at {adapter_path}")

# Training metrics from history
if trainer.state.log_history:
    # Get final evaluation loss
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    
    if eval_losses:
        print(f"\n📉 Final Evaluation Loss: {eval_losses[-1]:.4f}")
        print(f"   Best Evaluation Loss: {min(eval_losses):.4f}")
        print(f"   Initial Evaluation Loss: {eval_losses[0]:.4f}")
        print(f"   Improvement: {((eval_losses[0] - eval_losses[-1]) / eval_losses[0] * 100):.2f}%")
    
    if train_losses:
        print(f"\n📈 Final Training Loss: {train_losses[-1]:.4f}")
        print(f"   Best Training Loss: {min(train_losses):.4f}")
        print(f"   Initial Training Loss: {train_losses[0]:.4f}")

# Training speed
total_samples = len(train_dataset) * sft_config.num_train_epochs
samples_per_second = total_samples / total_training_time
print(f"\n⚡ Training Speed:")
print(f"   {samples_per_second:.2f} samples/second")
print(f"   {total_samples / total_training_time * 60:.2f} samples/minute")

# GPU utilization info
if torch.cuda.is_available():
    print(f"\n🎮 GPU Information:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"   Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

print("\n" + "="*70)

# 6. Save Metrics to JSON
# ---------------------------------------------------------
metrics = {
    "model_id": MODEL_ID,
    "adapter_type": "LoRA",
    "training_config": {
        "rank": peft_config.r,
        "alpha": peft_config.lora_alpha,
        "dropout": peft_config.lora_dropout,
        "batch_size": sft_config.per_device_train_batch_size,
        "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
        "effective_batch_size": sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps,
        "learning_rate": sft_config.learning_rate,
        "num_epochs": sft_config.num_train_epochs,
        "max_seq_length": sft_config.max_seq_length,
    },
    "dataset": {
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "total_samples": len(full_dataset),
    },
    "results": {
        "training_time_seconds": round(total_training_time, 2),
        "training_time_formatted": f"{hours}h {minutes}m {seconds}s",
        "adapter_size_mb": round(adapter_size_mb, 2) if os.path.exists(adapter_path) else None,
        "final_eval_loss": round(eval_losses[-1], 4) if eval_losses else None,
        "best_eval_loss": round(min(eval_losses), 4) if eval_losses else None,
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
        "samples_per_second": round(samples_per_second, 2),
    },
    "hardware": {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 2) if torch.cuda.is_available() else None,
    }
}

metrics_file = f"{new_model_name}/training_metrics.json"
with open(metrics_file, 'w') as f:
    json.dump(metrics, indent=2, fp=f)

print(f"\n✅ Metrics saved to: {metrics_file}")

# 7. Create a simple metrics visualization
# ---------------------------------------------------------
print("\n📊 Creating loss visualization...")

try:
    import matplotlib.pyplot as plt
    
    # Extract losses with their steps
    eval_data = [(log.get('step', i), log['eval_loss']) 
                 for i, log in enumerate(trainer.state.log_history) if 'eval_loss' in log]
    train_data = [(log.get('step', i), log['loss']) 
                  for i, log in enumerate(trainer.state.log_history) if 'loss' in log]
    
    if eval_data or train_data:
        plt.figure(figsize=(12, 6))
        
        if train_data:
            steps, losses = zip(*train_data)
            plt.plot(steps, losses, label='Training Loss', alpha=0.7, linewidth=2)
        
        if eval_data:
            steps, losses = zip(*eval_data)
            plt.plot(steps, losses, label='Validation Loss', alpha=0.7, linewidth=2, marker='o')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = f"{new_model_name}/loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        print(f"✅ Loss curve saved to: {plot_path}")
        plt.close()
except ImportError:
    print("⚠️  matplotlib not available, skipping visualization")
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")

print("\n🎉 Training complete with full metrics!")