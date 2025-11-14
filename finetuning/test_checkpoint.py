import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# --- 1. Configuration ---
base_model_id = "deepseek-ai/deepseek-coder-1.3b-instruct" # Or whichever model you used
adapter_path = "results/checkpoint-1000" # Replace with your checkpoint folder path
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# --- 4. Prepare Test Prompt ---
function_to_test = """
def calculate_area(length, width):
    \"\"\"Calculates the area of a rectangle.\"\"\"
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width
"""

# Use the EXACT same format as training
prompt = f"<s>[INST] Write a Python unit test for the following function:\n\n```python\n{function_to_test.strip()}\n```\n[/INST]\n"

print("\n--- Prompt ---")
print(prompt)

# --- 5. Tokenize ---
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

# --- 6. Generate ---
print("\n--- Generating Test ---")
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

# --- 7. Decode ---
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Generated Output ---")
# Often the output includes the prompt, let's try to isolate the generated part
# Find the end of the instruction marker `[/INST]` and print everything after it.
inst_end_marker = "[/INST]"
inst_end_index = generated_text.find(inst_end_marker)
if inst_end_index != -1:
    generated_test = generated_text[inst_end_index + len(inst_end_marker):].strip()
    print(generated_test)
else:
    # If marker not found, print the whole thing (might need adjustment)
    print(generated_text)

print("\n--- Done ---")