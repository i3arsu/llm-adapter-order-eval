import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datetime
import json

# 1. Postavke
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./tiny-retail-adapter-v2" # Pazi da je ovo točna putanja novog adaptera
start_time = datetime.datetime.now()

print(f"--- Početak testa: {start_time} ---")

# 2. Konfiguracija za RTX 3070 (4-bit učitavanje)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
)

# 3. Učitavanje Baznog Modela
print("Učitavam bazni model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4. Spajanje LoRA Adaptera (OVO JE NEDOSTAJALO)
print(f"Spajam LoRA adapter iz {adapter_path}...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 5. Funkcija za predikciju (Ažuriran format!)
def predict_intent(user_input):
    # BITNO: Format mora biti identičan onom iz treninga (TinyLlama Chat format)
    prompt = f"""<|system|>
Analyze the user request and extract action, product, and quantity into JSON format.</s>
<|user|>
{user_input}</s>
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,   # Greedy decoding za konzistentnost
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parsiranje odgovora (uzimamo sve nakon assistant taga)
    try:
        response_part = full_output.split("<|assistant|>\n")[-1].strip()
    except:
        response_part = full_output # Fallback ako split ne uspije
        
    return response_part

# 6. Prošireni set promptova (10 primjera)
test_sentences = [
    # Standardni add/remove
    "I want to remove 3 apples from my cart.",
    "Please add 5 bottles of water.",
    
    # Slang i neformalni govor
    "Nix the bananas.",
    "Can you toss 2 pizza into the basket?",
    "Grab me a sweater.",  # Implicitna količina (1)
    
    # Promjena mišljenja / Negacije
    "Actually, make that 10 eggs instead.",
    "Forget about the milk.",
    "I changed my mind, take out the jeans.",
    
    # Malo kompleksniji zahtjevi
    "Drop 1 blender from the order.",
    "I need 4 more notebooks please."
]

results = []

print("\n--- Rezultati Testiranja ---\n")
for sentence in test_sentences:
    result = predict_intent(sentence)
    
    # Validacija JSON-a (samo vizualno za log)
    try:
        json.loads(result)
        valid_json = "✅ Valid JSON"
    except:
        valid_json = "❌ Invalid JSON"

    print(f"Input:  {sentence}")
    print(f"Output: {result}")
    print(f"Status: {valid_json}")
    print("-" * 30)
    results.append((sentence, result))

# Spremanje rezultata
end_time = datetime.datetime.now()
duration = end_time - start_time
print(f"Ukupno trajanje: {duration}")

with open("test_results.txt", "w", encoding="utf-8") as file:
    file.write(f"Test run: {start_time}\n")
    file.write(f"Model: {adapter_path}\n")
    file.write("=" * 40 + "\n\n")
    for sentence, result in results:
        file.write(f"User Input: {sentence}\n")
        file.write(f"Model Output: {result}\n")
        file.write("-" * 30 + "\n")
    file.write(f"\nTotal duration: {duration}")