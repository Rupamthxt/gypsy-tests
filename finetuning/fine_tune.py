import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# *** MODIFIED: A simpler function to format a SINGLE example ***
# This is more reliable than the batch-based formatting function.
def format_training_example(example):
    # This will be applied to each row of the dataset
    text = f"PROMPT: {example['prompt']}\n\nCOMPLETION: {example['completion']}"
    # The map function expects a dictionary as a return value
    return {"text": text}


def run_finetuning():
    # --- 1. Configuration ---
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = "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"

    if not hf_token:
        print("Hugging Face token not found. Make sure you are logged in or have a .env file.")
        
    model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    dataset_file = "training_dataset.jsonl"
    new_model_name = "codegemma-2b-test-generator"
    
    # --- 2. Load the Dataset ---
    print(f"Loading dataset from {dataset_file}...")
    dataset = load_dataset("json", data_files=dataset_file, split="train", streaming=True)

    # *** NEW: Pre-process the dataset before training ***
    # We apply our formatting function to create a new "text" column.
    print("Formatting dataset...")
    formatted_dataset = dataset.map(format_training_example)

    # --- 3. Quantization Configuration (for QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- 4. Load Model and Tokenizer ---
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # --- 5. PEFT (LoRA) Configuration ---
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # --- 6. Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_steps=50,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=1000,
        warmup_ratio=0.03,
        lr_scheduler_type="constant"
    )

    # --- 7. Create the Trainer ---
    # *** MODIFIED: Stripped down to the bare minimum arguments ***
    # This is to ensure compatibility with your specific TRL version.
    # The trainer will infer the tokenizer and text field from the model and dataset.
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        args=training_args,
    )

    # --- 8. Start Training ---
    print("Starting the fine-tuning process...")
    trainer.train()

    # --- 9. Save the Final Model ---
    print(f"Training complete. Saving the fine-tuned model to '{new_model_name}'")
    trainer.model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    print("Model saved successfully!")
    
    # --- 10. (Optional) Test the Model ---
    print("\n--- Testing the fine-tuned model ---")
    from transformers import pipeline

    test_code = """
def calculate_area(length, width):
    \"\"\"Calculates the area of a rectangle.\"\"\"
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width
"""
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    
    # The test prompt now MUST match the format we used in training
    result = pipe(f"PROMPT: {test_code}\n\nCOMPLETION:")
    print(result[0]['generated_text'])
    print("---------------------------------")


if __name__ == "__main__":
    run_finetuning()

