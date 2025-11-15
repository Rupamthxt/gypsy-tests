from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "../checkpoints/merged_deepseek_coder",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("../checkpoints/merged_deepseek_coder")

model.save_pretrained("../checkpoints/merged_deepseek_coder_4bit", safe_serialization=True)
tokenizer.save_pretrained("../checkpoints/merged_deepseek_coder_4bit")
