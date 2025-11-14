from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# === CHANGE THESE ===
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"   # or your local base model
LORA_ADAPTER = "results/checkpoint-1000"                        # path to your fine-tuned LoRA adapter
OUTPUT_DIR = "./merged_deepseek_coder"                 # where to save merged model

# Load base model and tokenizer
print("🔹 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load LoRA adapter
print("🔹 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER)

# Merge LoRA weights into base model
print("🔹 Merging LoRA weights into base model...")
model = model.merge_and_unload()

# Save merged model + tokenizer
print("💾 Saving merged model...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Merged model saved to: {OUTPUT_DIR}")
