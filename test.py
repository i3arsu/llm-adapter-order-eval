import torch
import json
import pandas as pd
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))

# --- POSTAVKE ---
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapters_dir = "./adapters"
input_csv_file = "shopping_cart_final_normalized.csv" # Tvoj ulazni file
output_dir = "./results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Konfiguracija (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Učitavanje Modela
print(f"Učitavam model: {base_model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Pronalaženje svih "-merged" adaptera
print(f"Tražim -merged adaptere u '{adapters_dir}'...")
available_adapters = []
if os.path.exists(adapters_dir):
    for folder in os.listdir(adapters_dir):
        folder_path = os.path.join(adapters_dir, folder)
        if os.path.isdir(folder_path) and "-merged" in folder:
            available_adapters.append(folder_path)
            print(f"  Pronađen adapter: {folder}")

if not available_adapters:
    print(f"GREŠKA: Nema -merged adaptera pronađenih u '{adapters_dir}'!")
    exit()

print(f"Ukupno pronađeno {len(available_adapters)} -merged adapter(a).")

# 4. Funkcija za predikciju
def predict_intent(user_input, model):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Analyze the user request and extract action, product, and quantity into JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Čišćenje outputa (uzimamo samo JSON dio)
    try:
        response_part = full_output.split("assistant")[-1].strip()
        if "{" in response_part:
            start_idx = response_part.find("{")
            end_idx = response_part.rfind("}") + 1
            response_part = response_part[start_idx:end_idx]
    except:
        pass 
        
    return response_part

# 5. Učitavanje Datasetsa iz CSV-a
print(f"Učitavam pitanja iz '{input_csv_file}'...")
if not os.path.exists(input_csv_file):
    print("GREŠKA: Input CSV nije pronađen!")
    exit()

# Učitavamo bez headera jer tvoj file nema header (prvi stupac je tekst)
df_input = pd.read_csv(input_csv_file, header=None)
# Pretpostavljamo da je tekst pitanja u prvom stupcu (indeks 0)
test_sentences = df_input[0].tolist()

print(f"Ukupno pronađeno {len(test_sentences)} primjera.\n")

# --- TESTIRANJE ZA SVAKI ADAPTER ---
for adapter_path in available_adapters:
    adapter_name = os.path.basename(adapter_path)
    output_csv_file = os.path.join(output_dir, f"test_results_{adapter_name}.csv")
    
    print(f"\n{'='*60}")
    print(f"Pokrećem testiranje za adapter: {adapter_name}")
    print(f"{'='*60}\n")
    
    # Učitavamo adapter
    print(f"Spajam adapter: {adapter_path}...")
    try:
        current_model = PeftModel.from_pretrained(model, adapter_path)
        current_model.eval()
    except Exception as e:
        print(f"GREŠKA pri učitavanju {adapter_path}: {e}")
        continue
    
    # Lista za spremanje rezultata
    data_for_csv = []
    
    print("Pokrećem Testiranje...\n")
    
    for i, sentence in enumerate(test_sentences):
        # Ispis napretka svakih 50 primjera da znaš da radi
        if (i + 1) % 50 == 0:
            print(f"Obrađujem {i+1}/{len(test_sentences)}...")
    
        raw_response = predict_intent(sentence, current_model)
        
        # Pokušaj parsiranja JSON-a da ga razbijemo u stupce
        action = ""
        product = ""
        quantity = ""
        
        try:
            parsed = json.loads(raw_response)
            action = parsed.get("action", "")
            product = parsed.get("product", "")
            quantity = parsed.get("quantity", "")
        except json.JSONDecodeError:
            # Ako je JSON neispravan, ostavljamo prazno ili upisujemo error
            action = "ERROR"
            
        # Dodavanje u listu za CSV (samo traženi stupci)
        data_for_csv.append({
            "User Input": sentence,
            "llm_action": action,
            "llm_product": product,
            "llm_quantity": quantity,
        })
    
    # Spremanje u CSV pomoću pandasa
    df = pd.DataFrame(data_for_csv)
    df.to_csv(output_csv_file, index=False, encoding='utf-8')
    
    print(f"\nGotovo! Rezultati za '{adapter_name}' spremljeni u '{output_csv_file}'\n")