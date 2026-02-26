import torch
import json
import pandas as pd
import os
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))

# --- POSTAVKE ---
adapters_dir = "./adapters"
input_csv_file = "shopping_cart_final_normalized.csv" # Tvoj ulazni file
output_dir = "./results"

# Napomena: Koristi se "-merged" verzija gdje je adapter već spojen s base modelom

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Konfiguracija (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Pronalaženje svih "-merged" adaptera
print(f"Tražim -merged adaptere u '{adapters_dir}'...")
available_adapters = []
if os.path.exists(adapters_dir):
    for folder in os.listdir(adapters_dir):
        folder_path = os.path.join(adapters_dir, folder)
        # Tražimo samo "-merged" adaptere
        if os.path.isdir(folder_path) and "-merged" in folder and folder != "old":
            available_adapters.append(folder_path)
            print(f"  Pronađen merged adapter: {folder}")

if not available_adapters:
    print(f"GREŠKA: Nema -merged adaptera pronađenih u '{adapters_dir}'!")
    exit()

print(f"Ukupno pronađeno {len(available_adapters)} merged adapter(a).\n")

# 3. Funkcija za predikciju
def predict_intent(user_input, model, tokenizer):
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

# 4. Učitavanje Datasetsa iz CSV-a
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
    
    # Create individual folder for each adapter in results
    adapter_result_dir = os.path.join(output_dir, adapter_name)
    os.makedirs(adapter_result_dir, exist_ok=True)
    
    output_csv_file = os.path.join(adapter_result_dir, "test_results.csv")
    metadata_file = os.path.join(adapter_result_dir, "metadata.json")
    
    print(f"\n{'='*60}")
    print(f"Pokrećem testiranje za adapter: {adapter_name}")
    print(f"{'='*60}\n")
    
    # Učitavamo merged model (adapter + base model su već spojeni)
    print(f"Učitavam merged model: {adapter_path}...")
    try:
        current_model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        current_model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Model i tokenizer učitani")
    except Exception as e:
        print(f"GREŠKA pri učitavanju {adapter_path}: {e}")
        continue
    
    # Lista za spremanje rezultata
    data_for_csv = []
    
    print("Pokrećem Testiranje...\n")
    
    # Track timing and statistics
    test_start_time = datetime.now()
    errors_count = 0
    success_count = 0
    save_interval = 50  # Save every 50 items
    
    for i, sentence in enumerate(test_sentences):
        # Ispis napretka svakih 50 primjera da znaš da radi
        if (i + 1) % 50 == 0:
            print(f"Obrađujem {i+1}/{len(test_sentences)}...")
    
        try:
            raw_response = predict_intent(sentence, current_model, tokenizer)
            
            # Pokušaj parsiranja JSON-a da ga razbijemo u stupce
            action = ""
            product = ""
            quantity = ""
            parsing_status = "success"
            
            try:
                parsed = json.loads(raw_response)
                action = parsed.get("action", "")
                product = parsed.get("product", "")
                quantity = parsed.get("quantity", "")
                success_count += 1
            except json.JSONDecodeError:
                # Ako je JSON neispravan, ostavljamo prazno ili upisujemo error
                action = "ERROR"
                parsing_status = "failed"
                errors_count += 1
                
            # Dodavanje u listu za CSV (samo traženi stupci)
            data_for_csv.append({
                "User Input": sentence,
                "llm_action": action,
                "llm_product": product,
                "llm_quantity": quantity,
                "Parsing Status": parsing_status,
                "Raw Response": raw_response[:500],  # First 500 chars
            })
            
            # Save incrementally every N items to avoid data loss on crash
            if (i + 1) % save_interval == 0:
                df_checkpoint = pd.DataFrame(data_for_csv)
                df_checkpoint.to_csv(output_csv_file, index=False, encoding='utf-8')
                print(f"  → Checkpoint: Rezultati do reda {i+1} spremljeni")
                
        except Exception as e:
            print(f"GREŠKA pri obradi reda {i+1}: {str(e)}")
            # Dodaj redak greške i nastavi s testiranjem
            data_for_csv.append({
                "User Input": sentence,
                "llm_action": "ERROR",
                "llm_product": "N/A",
                "llm_quantity": "N/A",
                "Parsing Status": "exception",
                "Raw Response": f"Exception: {str(e)}",
            })
            errors_count += 1
            continue
    
    test_end_time = datetime.now()
    test_duration = (test_end_time - test_start_time).total_seconds()
    
    # Final save
    df = pd.DataFrame(data_for_csv)
    df.to_csv(output_csv_file, index=False, encoding='utf-8')
    print(f"✓ Finalni rezultati spremljeni")
    
    # Save metadata
    metadata = {
        "adapter_name": adapter_name,
        "adapter_path": adapter_path,
        "test_timestamp": test_start_time.isoformat(),
        "model_type": "merged (adapter + base model)",
        "total_tests": len(test_sentences),
        "successful_parses": success_count,
        "failed_parses": errors_count,
        "success_rate": f"{(success_count/len(test_sentences)*100):.2f}%" if test_sentences else "N/A",
        "test_duration_seconds": test_duration,
        "avg_time_per_test": f"{(test_duration/len(test_sentences)):.4f}" if test_sentences else "N/A"
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGotovo! Rezultati za '{adapter_name}' spremljeni u '{adapter_result_dir}'")
    print(f"  ✓ CSV rezultati: test_results.csv")
    print(f"  ✓ Metapodaci: metadata.json")
    print(f"  ✓ Uspješno parsiranih: {success_count}/{len(test_sentences)} ({(success_count/len(test_sentences)*100):.2f}%)")
    print(f"  ✓ Vrijeme izvršavanja: {test_duration:.2f}s\n")
    
    # Oslobodi memoriju nakon što se završi testiranje ovog adaptera
    del current_model
    del tokenizer
    torch.cuda.empty_cache()
    print(f"✓ Memorija oslobođena za sljedeći adapter\n")

print(f"\n{'='*60}")
print("TESTIRANJE ZAVRŠENO!")
print(f"{'='*60}")
print(f"Svi rezultati su spremljeni u '{output_dir}' folder")
print(f"Svaki adapter ima svoju podfolder sa:")
print(f"  - test_results.csv (detaljni rezultati)")
print(f"  - metadata.json (statistika i metapodaci)")
print(f"{'='*60}\n")