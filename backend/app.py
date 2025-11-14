import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
from fastapi import FastAPI, requests
from pydantic import BaseModel
import re

app = FastAPI()

# --- 1. Configuration ---
base_model_id = "deepseek-ai/deepseek-coder-1.3b-instruct" # Or whichever model you used
adapter_path = "../results/checkpoint-1000" # Replace with your checkpoint folder path
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Using CPU")
# --- 2. Load Base Model and Tokenizer (in 4-bit) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically uses GPU if available
    trust_remote_code=True,
    # low_cpu_mem_usage=True # Add if you face RAM issues loading
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
# Set padding token if it's not already set (common issue)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 3. Load the PEFT Adapter ---
print(f"Loading adapter: {adapter_path}")
# Ensure the adapter path exists
if not os.path.exists(adapter_path):
     raise FileNotFoundError(f"Adapter path not found: {adapter_path}. Make sure the path is correct.")

try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() # Optional: Merge adapter for faster inference, requires more RAM/VRAM initially
    print("Adapter loaded successfully.")
except Exception as e:
    print(f"Error loading adapter: {e}")
    # Fallback or exit if adapter loading fails
    # Maybe try loading without merge_and_unload first if memory is an issue
    # model = PeftModel.from_pretrained(base_model, adapter_path)


model.eval() # Set the model to evaluation mode

# Code for loading merged model directly

# model_path = "../merged_deepseek_coder"
# model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_path)



def extract_code_and_comment(model_output: str) -> str:
    """
    Extracts the Python code block and the last comment line (if any)
    from the model output.
    """
    # 1. Extract the first ```python ... ``` block
    code_match = re.search(r"```python(.*?)```", model_output, re.DOTALL)
    print(code_match)
    code = code_match.group(1).strip() if code_match else ""
    print(code)
    # 2. Extract the last line starting with 'Please note' or a comment-like line
    comment_match = re.search(r"(Please note.*|#.*)$", model_output.strip(), re.MULTILINE)
    comment = "# " + comment_match.group(1).strip() if comment_match else ""

    # 3. Combine them
    final_output = code
    if comment:
        final_output += "\n\n" + comment

    return final_output


class CodeInput(BaseModel):
    code: str

@app.post("/generate-tests")
async def generate_tests(req: CodeInput):
    prompt = f"<s>[INST] Write a Python unit test for the following function:\n\n```python\n{req.code.strip()}\n```\n[/INST]\n"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad(): # No need to calculate gradients for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,  # Adjust max length as needed
            do_sample=True,      # Use sampling for potentially more creative/varied tests
            temperature=0.6,     # Lower temperature for more focused output
            top_p=0.9,           # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id # Set pad token id
        )

    # # --- 7. Decode ---
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Often the output includes the prompt, let's try to isolate the generated part
    # Find the end of the instruction marker `[/INST]` and print everything after it.
    inst_end_marker = "[/INST]"
    inst_end_index = generated_text.find(inst_end_marker)
    if inst_end_index != -1:
        generated_test = generated_text[inst_end_index + len(inst_end_marker):].strip()
        print(generated_test)
    else:
        print(generated_text)
    clean_code = extract_code_and_comment(generated_test)
    return {"tests": clean_code}