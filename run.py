import pandas as pd
import torch
import datetime
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig 

start_time = datetime.datetime.now()
print(f"Vrijeme početka: {start_time}")

# --- KONFIGURACIJA ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./lora-llama3.1-retail"


# 1. Priprema podataka (Isto kao prije)
df = pd.read_csv("Retail_Dataset_synthetic.csv")
df = df.tail(5000).reset_index(drop=True)

def format_instruction_llama3(row):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Analyze the user request and extract action, product, and quantity into JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>

{row['user_input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{
    "action": "{row['action']}",
    "product": "{row['product']}",
    "quantity": {row['quantity']}
}}<|eot_id|>"""

df['text'] = df.apply(format_instruction_llama3, axis=1)
dataset = Dataset.from_pandas(df[['text']])

# 2. BitsAndBytes Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 3. Model i Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)

model = prepare_model_for_kbit_training(model)

# 4. LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)


# 5. Trening
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=25,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    dataloader_num_workers=0,
    report_to="none",
    packing=False
)

trainer = SFTTrainer(
    model=model,                  # Šaljemo "običan" model
    train_dataset=dataset,
    peft_config=peft_config,      # Šaljemo config, Trainer će sam napraviti PeftModel
    processing_class=tokenizer,
    args=sft_config
)

print("Počinjem trening Llama 3.1 (QLoRA)...")
trainer.train()

# 6. Spremanje
new_model_name = "llama3.1-retail-adapter"
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"Gotovo! Adapter spremljen u: {new_model_name}")
end_time = datetime.datetime.now()
print(f"Vrijeme završetka: {end_time}")
print(f"Ukupno trajanje: {end_time - start_time}")