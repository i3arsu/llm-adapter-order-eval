import torch
import datetime
import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- POSTAVKE ---
# Koristimo Llama 3 jer je to bio zadnji model treniran na 3070
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "./adapters/llama3-retail-3070-final" # Putanja iz zadnjeg treninga

start_time = datetime.datetime.now()
print(f"--- Početak testa: {start_time} ---")

# 1. Konfiguracija (4-bit za 8GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Učitavanje Modela
print(f"Učitavam bazni model: {base_model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Spajanje Adaptera
print(f"Spajam LoRA adapter iz: {adapter_path}...")
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
except Exception as e:
    print(f"GREŠKA kod učitavanja adaptera: {e}")
    print("Provjeri je li putanja do adaptera točna!")
    exit()

# 4. Funkcija za predikciju (Llama 3 Format)
def predict_intent(user_input):
    # Prompt mora biti IDENTIČAN onom iz treninga (Llama 3.1 format)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Analyze the user request and extract action, product, and quantity into JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,   # Greedy decoding za najbolji JSON
            pad_token_id=tokenizer.eos_token_id,
            # temperature=0.1 # Opcionalno: ako model halucinira, otkomentiraj ovo
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parsiranje: Tražimo zadnji 'assistant' header
    try:
        # Llama 3 output ponekad sadrži system prompt u decodeu, režemo sve do assistant dijela
        response_part = full_output.split("assistant")[-1].strip()
        # Ponekad ostane artifacta, tražimo prvu zagradu {
        if "{" in response_part:
            start_idx = response_part.find("{")
            end_idx = response_part.rfind("}") + 1
            response_part = response_part[start_idx:end_idx]
    except:
        pass # Vraća raw output ako faila
        
    return response_part

# 5. Prošireni Dataset (35 primjera)
test_sentences = [
    # --- Basic Add ---
    "Please add 5 bottles of water.",
    "Put 3 apples in the cart.",
    "Include 2 packs of coffee.",
    "I want to buy 10 notebooks.",
    "Throw in a chocolate bar.",

    # --- Basic Remove ---
    "I want to remove 3 apples from my cart.",
    "Take out the milk.",
    "Delete 2 bananas.",
    "Remove the headphones from my order.",
    "I don't need the socks anymore.",

    # --- Slang & Verbs ---
    "Nix the bananas.",
    "Can you toss 2 pizza into the basket?",
    "Grab me a sweater.",
    "Drop the eggs completely.",
    "Lose the shoes.",
    "Score me 3 energy drinks.",

    # --- Implicit Quantities ---
    "Add a watermelon.",
    "Remove the laptop.",
    "Get me an umbrella.",
    "I'd like a coke.",
    "Cancel the order of bread.",

    # --- Corrections / Complex ---
    "Actually, make that 10 eggs instead.",
    "Forget about the milk.",
    "I changed my mind, take out the jeans.",
    "Wait, I want 5 plates, not 2.",
    "Update the order, I need 2 more cables.",

    # --- Mixed / Tricky ---
    "Drop 1 blender from the order.",
    "I need 4 more notebooks please.",
    "Eliminate the orange juice.",
    "Zero out the pasta.",
    "Let's go with 6 yogurts.",
    "Pop 4 beers in the bag.",
    "Scrap the butter.",
    "Make sure I have 12 plates.",
    "Actually, remove everything."
]

results = []
valid_count = 0

print("\n--- Pokrećem Testiranje (Batch: 35) ---\n")

for i, sentence in enumerate(test_sentences):
    result = predict_intent(sentence)
    
    # Validacija JSON-a
    is_valid = False
    try:
        parsed = json.loads(result)
        # Provjera ključeva
        if "action" in parsed and "product" in parsed and "quantity" in parsed:
            is_valid = True
            valid_count += 1
            status_icon = "✅"
        else:
            status_icon = "⚠️ (Bad Keys)"
    except:
        status_icon = "❌ (Invalid JSON)"

    print(f"[{i+1}/{len(test_sentences)}] Input: {sentence}")
    print(f"Output: {result}")
    print(f"Status: {status_icon}")
    print("-" * 40)
    results.append((sentence, result, status_icon))

# Statistika
accuracy = (valid_count / len(test_sentences)) * 100
end_time = datetime.datetime.now()
duration = end_time - start_time

print(f"\n=== ZAVRŠNO IZVJEŠĆE ===")
print(f"Točnost JSON formata: {accuracy:.2f}% ({valid_count}/{len(test_sentences)})")
print(f"Ukupno trajanje: {duration}")

# Spremanje u datoteku
with open("test_results_detailed.txt", "w", encoding="utf-8") as file:
    file.write(f"Test run: {start_time}\n")
    file.write(f"Model: {base_model_id} + {adapter_path}\n")
    file.write(f"Accuracy: {accuracy:.2f}%\n")
    file.write("=" * 40 + "\n\n")
    for sentence, result, icon in results:
        file.write(f"In:  {sentence}\n")
        file.write(f"Out: {result}\n")
        file.write(f"Stat: {icon}\n")
        file.write("-" * 30 + "\n")