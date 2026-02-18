import json
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the CSV file
retail_dataset_queries = pd.read_csv("shopping_cart_final_normalized.csv")


def extract_json_object(text):
    # json_pattern = r'\{"action":\s*"[^"]+"\s*,\s*"product":\s*"[^"]+"\s*,\s*"quantity":\s*\d+\s*\}'
    json_pattern = r'\{\s*"action":\s*"[^"]+"\s*,\s*"product":\s*"[^"]+"\s*,\s*"quantity":\s*\d+\s*\}'

    
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {}


MODEL_IDS = [
    # "mistralai/Mistral-7B-v0.1",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    # "microsoft/phi-4", 
    "google/gemma-3-4b-pt", 
    # "ibm-granite/granite-3.3-8b-base", 
    # "meta-llama/Llama-3.1-8B", 
    # "meta-llama/Llama-3.2-3B", 
    # "Qwen/Qwen3-4B", 
    # "Qwen/Qwen3-8B"
    ]

SYSTEM_PROMPT = """
You are a shopping-cart assistant whose only job is to parse the user request and output a single JSON object with this exact schema:

{
"action":   "<add|remove>",
"product":  "<exact product name>",
"quantity": <integer>
}

Rules:
1. "action" must be either "add" or "remove". Map any synonyms ("put in", "insert", "take out", "nix", "delete", etc.) to these two.
2. "product" is exactly what the customer wants, stripped of any action words or numbers.
3. "quantity" is an integer. If the user does not specify a number, default to 1.
4. Output ONLY the JSON - no markdown, no explanations, no extra keys or text.

Examples:

User: "Please put 3 cans of soda into my cart."
Output:
{"action":"add","product":"cans of soda","quantity":3}

User: "Nix 2 backpacks"
Output:
{"action":"remove","product":"backpacks","quantity":2}

User: "Add apples"
Output:
{"action":"add","product":"apples","quantity":1}

Now parse the user next message.
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def build_prompt(user_input):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_response(user_input, model, tokenizer):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        response_part = full_output.split("assistant")[-1].strip()
        if "{" in response_part:
            start_idx = response_part.find("{")
            end_idx = response_part.rfind("}") + 1
            response_part = response_part[start_idx:end_idx]
    except Exception:
        response_part = full_output

    return response_part

print(f"Processing {len(retail_dataset_queries)} rows...")
print("Starting sequential processing...")

for model_id in MODEL_IDS:
    results = []
    print(f"Processing with model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id)

    for index, row in retail_dataset_queries.iterrows():
        print(f"Processing row {index + 1}/{len(retail_dataset_queries)}")
        response = get_response(row["user_input"], model, tokenizer)
    
        # Assuming the response is a JSON string, parse it
        try:
            response_data = extract_json_object(response)
            # response_data = json.loads(response)
            llm_action = response_data.get("action", "")
            llm_product = response_data.get("product", "")
            llm_quantity = response_data.get("quantity", 0)

            result = {
                "raw_response": response,
                "llm_action": llm_action,
                "llm_product": llm_product,
                "llm_quantity": llm_quantity,
            }

            results.append(result)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row {index}: {response}")
            print(f"JSON decode error: {e}")
            # continue
            result = {
                "raw_response": response,
                "llm_action": "",
                "llm_product": "",
                "llm_quantity": 0,
            }
            results.append(result)
            continue


    results_df = pd.DataFrame(results)
    sanitized_model_name = re.sub(r'[/:.]', '_', model_id)
    output_filename = f"Retail_Dataset_LLM_Responses_{sanitized_model_name}.csv"
    results_df.to_csv(output_filename, index=False)
