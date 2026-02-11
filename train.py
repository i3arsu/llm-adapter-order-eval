import os
import torch
import pandas as pd
import logging
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

# Postavljanje logginga
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimizacija memorije za MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def train_with_validation_mac():
    # 1. Provjera Hardvera
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Koristim Apple MPS (GPU acceleration)")
    else:
        device = "cpu"
        print("⚠️ MPS nije dostupan, koristim CPU.")

    # --- KONFIGURACIJA ---
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = "./adapters/tiny-retail-val-mps"

    # 2. Priprema i Čišćenje Podataka
    print("🧹 Učitavam i čistim podatke...")
    df = pd.read_csv("Retail_Dataset_10000.csv")
    
    # Funkcija za provjeru kvalitete (odbacuje neispravne retke)
    def is_data_valid(row):
        return str(row['product']).lower() in str(row['user_input']).lower()

    initial_count = len(df)
    df = df[df.apply(is_data_valid, axis=1)]
    print(f"Očišćeno: {initial_count - len(df)} redaka. Ostalo: {len(df)} redaka.")

    # Uzimamo uzorak ako je dataset prevelik za brzi test (npr. 5000 redaka)
    if len(df) > 5000:
        df = df.tail(5000).reset_index(drop=True)

    # Formatiranje za TinyLlama Chat (System/User/Assistant)
    def format_instruction(row):
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

    df['text'] = df.apply(format_instruction, axis=1)
    full_dataset = Dataset.from_pandas(df[['text']])

    # --- 80/20 SPLIT ---
    dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    
    print(f"📊 Trening set: {len(train_dataset)} | Validacijski set: {len(eval_dataset)}")

    # 3. Model i Tokenizer
    print(f"🧠 Učitavam model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Učitavanje u float16 (Nativno za M1/M2)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None # Isključujemo auto-mapiranje za MPS
    )
    model.to(device)
    
    # Štednja memorije
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Trening Argumenti s Validacijom
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # Više epoha jer je model mali
        per_device_train_batch_size=2,   # Konzervativno za 16GB RAM
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        
        # Postavke za optimizaciju na Macu
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        
        # Validacija
        eval_strategy="steps",
        eval_steps=50,                  # Evaluacija svakih 50 koraka
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,    # Vraća najbolji model na kraju
        metric_for_best_model="eval_loss",
        
        logging_steps=10,
        report_to="none",
        hub_token=None,
        push_to_hub=False,
    )
    
    tokenizer.model_max_length = 512
    
    trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
)

    print("🚀 Počinjem trening s validacijom...")
    trainer.train()

    # 6. Spremanje
    final_name = "tiny-retail-best-mps"
    print(f"💾 Spremanje najboljeg modela u: {final_name}")
    trainer.model.save_pretrained(final_name)
    tokenizer.save_pretrained(final_name)

if __name__ == "__main__":
    train_with_validation_mac()