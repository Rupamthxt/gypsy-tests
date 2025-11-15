from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "rupamthxt/deepseek-coder-1.3b-instruct-unittest"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "Write a Python test case for: def add(a, b): return a + b"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
