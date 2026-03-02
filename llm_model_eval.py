import json
import pandas as pd
import re
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the CSV file
retail_dataset_queries = pd.read_csv("shopping_cart_final_normalized.csv")


def extract_json_object(text):
    # Remove <think> tags and their content for DeepSeek-R1 models
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<｜begin▁of▁sentence｜>.*?<｜Assistant｜>', '', text, flags=re.DOTALL)
    
    # Try to find JSON object - more flexible pattern
    json_pattern = r'\{\s*"action":\s*"[^"]+"\s*,\s*"product":\s*"[^"]+"\s*,\s*"quantity":\s*\d+\s*\}'
    
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {}


MODEL_IDS = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "google/gemma-3-4b-it",
    "ibm-granite/granite-3.3-2b-instruct", 
    "meta-llama/Llama-3.1-8B-Instruct", 
    "meta-llama/Llama-3.2-3B-Instruct", 
    "Qwen/Qwen3-4B", 
    "Qwen/Qwen3-8B"
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
4. Output ONLY the JSON - no markdown, no explanations, no extra keys or text, no reasoning, no thinking process.
5. Do NOT use <think> tags or show your reasoning. Output the JSON directly.

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


def build_inputs(user_input, tokenizer):
    """Build tokenized inputs using the model's native chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
    except Exception as e:
        print(f"Warning: No chat template found, using simple concatenation. Error: {e}")
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\n\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt")
    
    return inputs


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # gemma-3-*-it models are multimodal and need a different class
    if "gemma-3" in model_id.lower() and "it" in model_id.lower():
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Gemma-3 uses a processor instead of tokenizer for chat template
        tokenizer = AutoProcessor.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def get_response(user_input, model, tokenizer, debug=False):
    inputs = build_inputs(user_input, tokenizer)
    input_len = inputs['input_ids'].shape[-1]
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # High limit for reasoning models; they'll stop at EOS naturally
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (after the prompt)
    new_tokens = outputs[0][input_len:]
    response_part = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    if debug:
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n--- DEBUG: Full output with tokens ---")
        print(full_output)
        print(f"\n--- DEBUG: Extracted response ---")
        print(response_part)
        print("---\n")

    return response_part

print(f"Processing {len(retail_dataset_queries)} rows...")
print("Starting sequential processing...")

for model_id in MODEL_IDS:
    results = []
    sanitized_model_name = re.sub(r'[/:.]', '_', model_id)
    output_filename = f"Retail_Dataset_LLM_Responses_{sanitized_model_name}.csv"
    
    print(f"\n{'='*60}")
    print(f"Processing with model: {model_id}")
    print(f"{'='*60}")
    model, tokenizer = load_model_and_tokenizer(model_id)

    for index, row in retail_dataset_queries.iterrows():
        if (index + 1) % 50 == 0:
            print(f"Processing row {index + 1}/{len(retail_dataset_queries)}...")
        
        # Debug first 3 rows per model
        debug = (index < 3)
        response = get_response(row["user_input"], model, tokenizer, debug=debug)
    
        response_data = extract_json_object(response)
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
        
        # Incremental save every 50 rows
        if (index + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(output_filename, index=False)

    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)
    print(f"✓ Results saved to {output_filename}")
    
    # Free GPU memory before loading next model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
