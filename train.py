import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig # <--- KORISTIMO SFTConfig

# --- KONFIGURACIJA ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./adapters/llama3-retail-adapter_v2"

login(token=os.getenv("HF_TOKEN"))

# 1. Priprema Podataka
# ---------------------------------------------------------
print("Učitavam i čistim podatke...")
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

print(f"Trening set: {len(train_dataset)} | Validacija: {len(eval_dataset)}")

# 2. Model i Config
# ---------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    attn_implementation="eager"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Trening Argumenti (SFTConfig umjesto TrainingArguments)
# ---------------------------------------------------------
# Ovdje definiramo sve: i parametre treninga i parametre dataseta
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # --- PARAMETRI DATASETA (PREBAČENO OVDJE) ---
    dataset_text_field="text",
    max_length=512,
    packing=False,
    # --------------------------------------------

    # --- MEMORY SAVING (RTX 3070) ---
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    # --------------------------------

    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=25,
    optim="paged_adamw_32bit",
    
    # Validacija (novo ime parametra)
    eval_strategy="steps", 
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config # Šaljemo novi SFTConfig
)

print("Počinjem optimizirani trening na RTX 3070...")
trainer.train()

# 4. Spremanje
new_model_name = "llama3-retail-3070-final"
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"Gotovo! Model spremljen u: {new_model_name}")