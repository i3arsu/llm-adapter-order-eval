import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Postavke - moraju biti iste kao kod treniranja
# ---------------------------------------------------------
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "./retail-adapter-v1" # Putanja gdje si spremio adapter

# 2. Učitavanje Baznog Modela
# ---------------------------------------------------------
# Ovdje ne moramo koristiti 4-bita ako imamo dovoljno memorije za inference,
# ali je sigurnije ostaviti ako testiraš na istom GPU-u.
# Ako želiš brži inference i imaš memorije, makni 'load_in_4bit=True'.
print("Učitavam bazni model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. Učitavanje i spajanje LoRA Adaptera
# ---------------------------------------------------------
print(f"Učitavam LoRA adapter iz {adapter_path}...")
model = PeftModel.from_pretrained(model, adapter_path)

# Prebacujemo model u mode za evaluaciju
model.eval()

# 4. Funkcija za testiranje
# ---------------------------------------------------------
def predict_intent(user_input):
    # Format mora biti IDENTIČAN onom iz treninga!
    prompt = f"""### Instruction:
Analyze the user request and extract action, product, and quantity.

### Input:
{user_input}

### Response:
"""
    
    # Tokenizacija
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generiranje odgovora
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, # Dovoljno za JSON odgovor
            do_sample=False,   # Deterministički ispis (bolje za JSON)
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Dekodiranje rezultata
    # 'outputs[0]' sadrži cijeli tekst (prompt + odgovor), pa moramo odrezati prompt
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Izdvajamo samo dio nakon "### Response:"
    response_part = full_output.split("### Response:\n")[-1].strip()
    return response_part

# 5. Primjeri za testiranje
# ---------------------------------------------------------
test_sentences = [
    "I want to remove 3 apples from my cart.",
    "Please add 5 bottles of water.",
    "Forget about the milk.",
    "Actually, make that 10 eggs.",
    "Can you toss 2 pizza into the basket?"
]

print("\n--- Rezultati Testiranja ---\n")
for sentence in test_sentences:
    result = predict_intent(sentence)
    print(f"User Input: {sentence}")
    print(f"Model Output: {result}")
    print("-" * 30)