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

print(f"Ukupno pronađeno {len(available_adapters)} merged adapter(a):\n")
for adapter in available_adapters:
    print(f"  ✓ {os.path.basename(adapter)}")
print()

# 3. Funkcija za predikciju
def predict_intent(user_input, model, tokenizer, debug=False):
    """
    Generate predictions using the model's native chat template.
    Works with any model that has tokenizer.apply_chat_template() support.
    """
    messages = [
        {"role": "system", "content": "Analyze the user request and extract action, product, and quantity into JSON format."},
        {"role": "user", "content": user_input}
    ]
    
    try:
        # Try using the model's built-in chat template (preferred method)
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
    except AttributeError:
        # Fallback for models without chat template support
        print(f"⚠️  Model doesn't have chat template, using manual formatting")
        prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nResponse:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2048, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if debug:
            print(f"\n[DEBUG] Full output:\n{repr(full_output)}\n")
        
        # Čišćenje outputa - extract only the first valid JSON object
        try:
            # Split by assistant marker to get the response part
            if "assistant" in full_output:
                response_part = full_output.split("assistant")[-1]
            else:
                response_part = full_output
            
            response_part = response_part.strip()
            
            # Extract the FIRST valid JSON object
            if "{" in response_part:
                start_idx = response_part.find("{")
                # Find matching closing brace (handle nested braces)
                depth = 0
                end_idx = start_idx
                for j in range(start_idx, len(response_part)):
                    if response_part[j] == "{":
                        depth += 1
                    elif response_part[j] == "}":
                        depth -= 1
                        if depth == 0:
                            end_idx = j + 1
                            break
                response_part = response_part[start_idx:end_idx]
        except:
            pass 
            
        return response_part
    
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error during generation: {e}")
        return f"ERROR: {str(e)}"

# 4. Učitavanje Datasetsa iz CSV-a
print(f"Učitavam pitanja iz '{input_csv_file}'...")
if not os.path.exists(input_csv_file):
    print("GREŠKA: Input CSV nije pronađen!")
    exit()

# CSV has a header row: user_input,action,product,quantity
df_input = pd.read_csv(input_csv_file)
test_sentences = df_input['user_input'].tolist()

print(f"Ukupno pronađeno {len(test_sentences)} primjera.\n")

# --- TESTIRANJE ZA SVAKI ADAPTER ---
skipped_adapters = []
tested_adapters = []

for adapter_path in available_adapters:
    adapter_name = os.path.basename(adapter_path)
    
    # Create individual folder for each adapter in results
    adapter_result_dir = os.path.join(output_dir, adapter_name)
    os.makedirs(adapter_result_dir, exist_ok=True)
    
    output_csv_file = os.path.join(adapter_result_dir, "test_results.csv")
    metadata_file = os.path.join(adapter_result_dir, "metadata.json")
    
    # Check if adapter has already been fully tested
    if os.path.exists(metadata_file):
        print(f"\n{'='*60}")
        print(f"⊘ Adapter već je testiran: {adapter_name}")
        print(f"{'='*60}")
        print(f"Preskačem - rezultati postoje u: {adapter_result_dir}\n")
        skipped_adapters.append(adapter_name)
        continue
    
    tested_adapters.append(adapter_name)
    
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
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        current_model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Model i tokenizer učitani")
        print(f"  - Chat template available: {tokenizer.chat_template is not None}")
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
            # Enable debug for first 3 predictions
            debug_mode = (i < 3)
            raw_response = predict_intent(sentence, current_model, tokenizer, debug=debug_mode)
            
            # Pokušaj parsiranja JSON-a da ga razbijemo u stupce
            action = ""
            product = ""
            quantity = ""
            parsing_status = "success"
            
            try:
                parsed = json.loads(raw_response)
                # Handle nested JSON (e.g. Deepseek: {"user": {"action": ...}})
                if "action" not in parsed and len(parsed) == 1:
                    inner = list(parsed.values())[0]
                    if isinstance(inner, dict) and "action" in inner:
                        parsed = inner
                action = parsed.get("action", "")
                product = parsed.get("product", parsed.get("item", ""))
                quantity = parsed.get("quantity", "")
                if action and product:
                    success_count += 1
                else:
                    parsing_status = "partial"
                    errors_count += 1
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
print(f"\nTesrirani adapteri ({len(tested_adapters)}):")  # Shows which adapters were tested in this run
for adapter in tested_adapters:
    print(f"  ✓ {adapter}")

if skipped_adapters:
    print(f"\nPreskočeni adapteri - već su testirani ({len(skipped_adapters)}):")
    for adapter in skipped_adapters:
        print(f"  ⊘ {adapter}")

print(f"\nSvi rezultati su spremljeni u '{output_dir}' folder")
print(f"Svaki adapter ima svoju podfolder sa:")
print(f"  - test_results.csv (detaljni rezultati)")
print(f"  - metadata.json (statistika i metapodaci)")
print(f"{'='*60}\n")