import pandas as pd
import torch
import datetime
import os
from huggingface_hub import login
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig 
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN environment variable not set. You may need to authenticate.")

start_time = datetime.datetime.now()
print(f"🚀 Vrijeme početka: {start_time}")

hf_token = os.getenv("HF_TOKEN")
# --- KONFIGURACIJA ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = ".adapters/llama3-retail-colab"

# 1. Priprema podataka
df = pd.read_csv("Retail_Dataset_synthetic.csv")
df = df.tail(5000).reset_index(drop=True)

# POPRAVAK 1: Format prilagođen TinyLlami (ne Llama-3!)
def format_instruction_tiny(row):
    return f"""<|system|>
Analyze the user request and extract action, product, and quantity into JSON format.</s>
<|user|>
{row['user_input']}</s>
<|assistant|>
{{
    "action": "{row['action']}",
    "product": "{row['product']}",
    "quantity": {row['quantity']}
}}</s>"""

df['text'] = df.apply(format_instruction_tiny, axis=1)
dataset = Dataset.from_pandas(df[['text']])

# 2. BitsAndBytes Config (Optimizirano za RTX 30-series)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # RTX 3070 podržava bfloat16 (jako bitno za stabilnost!)
    bnb_4bit_use_double_quant=True,
)

# 3. Model i Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # SFTTrainer voli right padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager" # Možeš probati "flash_attention_2" ako instaliraš biblioteku, ali eager je ok
)

model = prepare_model_for_kbit_training(model)

# 4. LoRA Config
peft_config = LoraConfig(
    r=32,           # Povećan rank na 32 za bolje učenje (imaš memorije za to na 3070)
    lora_alpha=64,  # Alpha = 2 * r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Targetiramo sve linearne slojeve za bolju točnost
)

# 5. Trening Konfiguracija
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,
    num_train_epochs=3,             # POPRAVAK 2: Više epoha (1 -> 3)
    per_device_train_batch_size=8,  # POPRAVAK 3: Veći batch (imaš 8GB VRAM, TinyLlama je mala)
    gradient_accumulation_steps=2,  # Efektivni batch = 16
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                      # Koristimo bfloat16 na RTX 3070
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    report_to="none",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config
)

print("💪 Počinjem trening (TinyLlama Optimized)...")
trainer.train()

# 6. Spremanje
new_model_name = "tiny-retail-adapter-v2"
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"🎉 Gotovo! Adapter spremljen u: {new_model_name}")

end_time = datetime.datetime.now()
print(f"⏱️ Ukupno trajanje: {end_time - start_time}")