from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./merged_deepseek_coder"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "Write a Python test case for: def translate_text(text, target_language): translator = Translator() translation = translator.translate(text, dest=target_language) return translation.text"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model = AutoModelForCausalLM.from_pretrained(
#     "./merged_deepseek_coder",
#     load_in_4bit=True,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained("./merged_deepseek_coder")

# model.save_pretrained("./merged_deepseek_coder_4bit", safe_serialization=True)
# tokenizer.save_pretrained("./merged_deepseek_coder_4bit")
