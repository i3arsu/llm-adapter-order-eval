import os
import torch
import pandas as pd
import logging
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

# Konfiguracija logginga
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optimizacija memorije za MPS (Apple Silicon)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def train_model():
    logger.info("Inicijalizacija trening skripte na M1 Pro okruženju.")

    # Provjera hardvera
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info(f"Koristim uređaj: {device} (Metal Performance Shaders)")
    else:
        device = "cpu"
        logger.warning("MPS nije dostupan. Koristim CPU (ovo će biti sporo).")

    # KONFIGURACIJA
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = "./lora-retail-mps"
    
    # 1. Priprema podataka
    logger.info("Učitavanje i obrada podataka...")
    try:
        df = pd.read_csv("Retail_Dataset_synthetic.csv")
        # Uzimamo manji set radi demonstracije brzine, prilagoditi po potrebi
        df = df.tail(2000).reset_index(drop=True) 
        logger.info(f"Dataset učitan: {len(df)} redaka.")
    except Exception as e:
        logger.error(f"Greška kod učitavanja CSV-a: {e}")
        return

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
    dataset = Dataset.from_pandas(df[['text']])

    # 2. Model i Tokenizer
    logger.info(f"Učitavanje modela: {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Potrebno za SFTTrainer

    # Učitavanje u float16 (half precision) - optimalno za M1
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=None # Isključujemo auto mapiranje radi izbjegavanja grešaka s accelerateom na Macu
    )
    model.to(device)

    # Postavke za štednju memorije
    model.config.use_cache = False
    model.enable_input_require_grads() 
    
    # 3. LoRA Konfiguracija
    logger.info("Konfiguriranje LoRA adaptera...")
    peft_config = LoraConfig(
        r=8, # Manji rank za manju potrošnju memorije
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable_params} || All params: {all_params} || %: {100 * trainable_params / all_params:.4f}")

    # 4. Trening Argumenti
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1, # Batch size 1 je nužan za 16GB RAM-a
        gradient_accumulation_steps=4, # Kompenzacija za mali batch size
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no", # Ne spremamo checkpointove tijekom treninga radi prostora
        optim="adamw_torch", # Nativni PyTorch optimizator (bitno za MPS)
        fp16=False, # Isključeno jer MPS može imati problema s mixed precision u Traineru
        bf16=False, 
        use_mps_device=True, # Forsiranje MPS-a u Traineru
        report_to="none",
        dataloader_pin_memory=False, # Ponekad pomaže kod memory leakova na Macu
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # 5. Pokretanje treninga
    logger.info("Započinjem trening...")
    try:
        trainer.train()
        logger.info("Trening uspješno završen.")
    except Exception as e:
        logger.error(f"Greška tijekom treninga: {e}")
        return

    # 6. Spremanje
    new_model_name = "retail-adapter-mps"
    logger.info(f"Spremanje adaptera u: {new_model_name}")
    trainer.model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    logger.info("Proces završen.")

if __name__ == "__main__":
    train_model()