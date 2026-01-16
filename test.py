import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 1. Postavke - moraju biti iste kao kod treniranja
# ---------------------------------------------------------
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./retail-adapter-mps" # Path do adaptera

# Provjera dostupnosti MPS-a (GPU na Macu)
if torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Koristim Apple M1/M2 GPU (MPS acceleration)")
else:
    device = "cpu"
    print("⚠️ MPS nije dostupan, koristim CPU (ovo će biti sporije)")

# 2. Učitavanje Baznog Modela
# Učitavamo direktno u float16 (half precision) što je native za M1 čipove
print("Učitavam bazni model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16, 
    device_map=None # BITNO: Isključujemo auto mapiranje da izbjegnemo greške
)

# Ručno prebacujemo model na MPS uređaj
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. Učitavanje LoRA Adaptera
print(f"Spajam LoRA adapter iz {adapter_path}...")
model = PeftModel.from_pretrained(
    model, 
    adapter_path, 
    device_map=None # Također isključujemo auto mapiranje ovdje
)
model.to(device) # Osiguravamo da je i adapter na GPU
model.eval()

# 4. Funkcija za predikciju
def predict_intent(user_input):
    prompt = f"""### Instruction:
Analyze the user request and extract action, product, and quantity.

### Input:
{user_input}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.split("### Response:\n")[-1].strip()

# 5. Testiranje
sentences = [
    "I want to remove 3 apples from my cart.",
    "Please add 5 bottles of water.",
    "Actually, make that 10 eggs.",
]

print("\n--- Testiranje na M1 Pro ---")
start_time = time.time()

for s in sentences:
    res = predict_intent(s)
    print(f"Input: {s}")
    print(f"Output: {res}\n")

print(f"Vrijeme izvršavanja: {time.time() - start_time:.2f} sekundi")