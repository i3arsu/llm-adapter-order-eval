import torch
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training
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
MODEL_IDS = [
    "mistralai/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    "microsoft/phi-4", 
    "google/gemma-3-4b-pt", 
    "ibm-granite/granite-3.3-8b-base", 
    "meta-llama/Llama-3.1-8B", #
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen3-4B", 
    "Qwen/Qwen3-8B"
]

ADAPTER_TYPE = "ia3"  # Will be used in naming

# Auto-generate naming from model and config
def extract_model_name(model_id):
    # Extract short name like "llama3.1" from "meta-llama/Llama-3.1-8B-Instruct"
    name = model_id.split("/")[-1].lower()
    if "llama-3.1" in name or "llama3.1" in name:
        return "llama3.1"
    elif "llama-3" in name or "llama3" in name:
        return "llama3"
    elif "mistral" in name:
        return "mistral"
    elif "gemma" in name:
        return "gemma"
    else:
        return name.split("-")[0]

login(token=os.getenv("HF_TOKEN"))

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

# Main training loop for multiple models
# ---------------------------------------------------------
for MODEL_ID in MODEL_IDS:
    print(f"\n{'='*70}")
    print(f"Training model: {MODEL_ID}")
    print(f"{'='*70}\n")
    
    MODEL_SHORT_NAME = extract_model_name(MODEL_ID)
    OUTPUT_DIR = f"./{ADAPTER_TYPE}-{MODEL_SHORT_NAME}-checkpoints"

    # 2. Model and IA3 Config - OPTIMIZED FOR A100
    # ---------------------------------------------------------
    print("Loading model in BF16 precision (optimized for A100)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Native BF16 support on A100
        use_cache=False,
        attn_implementation="eager"  # A100 supports Flash Attention 2
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)  # Prepares model for quantization-aware training

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
    print("\n" + "="*70)
    model.print_trainable_parameters()
    print("="*70 + "\n")

    # 3. Training Configuration - A100 OPTIMIZED FOR IA3
    # ---------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        
        # --- DATASET PARAMETERS ---
        dataset_text_field="text",
        max_length=1024,
        packing=True,
        # --------------------------

        # --- A100 OPTIMIZED BATCH SETTINGS ---
        # IA3 has fewer parameters, so we can use larger batches
        per_device_train_batch_size=4,  # Doubled from LoRA version
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
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
        logging_dir=f"{OUTPUT_DIR}/logs",
        ddp_find_unused_parameters=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config
    )

    is_main_process = trainer.is_world_process_zero()

    print("Starting A100-optimized IA3 training...")
    print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"Total training steps: ~{len(train_dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps) * sft_config.num_train_epochs}")

    # Start timing
    training_start_time = time.time()

    # Train
    trainer.train()

    # Calculate training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # 4. Save Final Model
    # ---------------------------------------------------------
    if is_main_process:
        os.makedirs("./adapters", exist_ok=True)
        new_model_name = f"./adapters/{ADAPTER_TYPE}-{MODEL_SHORT_NAME}"
        trainer.model.save_pretrained(new_model_name)
        tokenizer.save_pretrained(new_model_name)
        print(f"Training complete! IA3 adapter saved to: {new_model_name}")

        # Optional: Save merged model
        print("\nMerging IA3 adapter with base model...")
        merged_model = trainer.model.merge_and_unload()
        merged_model_name = f"./adapters/{ADAPTER_TYPE}-{MODEL_SHORT_NAME}-merged"
        merged_model.save_pretrained(merged_model_name)
        tokenizer.save_pretrained(merged_model_name)
        print(f"Merged model saved to: {merged_model_name}")
    else:
        new_model_name = f"./adapters/{ADAPTER_TYPE}-{MODEL_SHORT_NAME}"

    # 5. Calculate and Display Metrics
    # ---------------------------------------------------------
    if is_main_process:
        print("\n" + "="*70)
        print("TRAINING METRICS SUMMARY")
        print("="*70)

    # Training time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    if is_main_process:
        print(f"\n📊 Training Time:")
        print(f"   Total: {total_training_time:.2f} seconds ({hours}h {minutes}m {seconds}s)")
        print(f"   Per Epoch: {total_training_time / sft_config.num_train_epochs:.2f} seconds")

    # Adapter size
    adapter_path = f"{new_model_name}/adapter_model.safetensors"
    if is_main_process:
        if os.path.exists(adapter_path):
            adapter_size_bytes = os.path.getsize(adapter_path)
            adapter_size_mb = adapter_size_bytes / (1024 * 1024)
            print(f"\n💾 Adapter Size:")
            print(f"   {adapter_size_mb:.2f} MB ({adapter_size_bytes:,} bytes)")
        else:
            print(f"\n⚠️  Adapter file not found at {adapter_path}")
            adapter_size_mb = None
    else:
        adapter_size_mb = None

    # Training metrics from history
    if is_main_process and trainer.state.log_history:
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
    if is_main_process:
        print(f"\n⚡ Training Speed:")
        print(f"   {samples_per_second:.2f} samples/second")
        print(f"   {total_samples / total_training_time * 60:.2f} samples/minute")

    # GPU utilization info
    if is_main_process and torch.cuda.is_available():
        print(f"\n🎮 GPU Information:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"   Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

    if is_main_process:
        print("\n" + "="*70)

    # 6. Save Metrics to JSON
    # ---------------------------------------------------------
    metrics = {
        "model_id": MODEL_ID,
        "adapter_type": "IA3",
        "training_config": {
            "target_modules": list(ia3_config.target_modules),
            "feedforward_modules": list(ia3_config.feedforward_modules),
            "batch_size": sft_config.per_device_train_batch_size,
            "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
            "effective_batch_size": sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps,
            "learning_rate": sft_config.learning_rate,
            "num_epochs": sft_config.num_train_epochs,
            "max_length": sft_config.max_length,
            "weight_decay": sft_config.weight_decay,
        },
        "dataset": {
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "total_samples": len(full_dataset),
        },
        "results": {
            "training_time_seconds": round(total_training_time, 2),
            "training_time_formatted": f"{hours}h {minutes}m {seconds}s",
            "adapter_size_mb": round(adapter_size_mb, 2) if adapter_size_mb else None,
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

    if is_main_process:
        metrics_file = f"{new_model_name}/training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, indent=2, fp=f)

        print(f"\n✅ Metrics saved to: {metrics_file}")

    # 7. Create a simple metrics visualization
    # ---------------------------------------------------------
    if is_main_process:
        print("\n📊 Creating loss visualization...")

    if is_main_process:
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

    if is_main_process:
        print(f"\n🎉 Training complete for {MODEL_SHORT_NAME} with full metrics!")
    
    # Clear GPU memory for next model
    del model, trainer, tokenizer
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("✅ All models trained successfully!")
print("="*70)