import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import os

def calculate_perplexity(model, tokenizer, dataset, device="cuda"):
    """Calculates perplexity for a given model and dataset, masking the prompt."""
    model.eval()
    total_nll = 0.0
    total_length = 0
    
    response_template = "<|im_start|>assistant\n"
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    response_len = len(response_token_ids)

    with torch.no_grad():
        for i, item in enumerate(dataset):
            text = item.get('text', '')
            if not text:
                if 'messages' in item:
                    text = tokenizer.apply_chat_template(item['messages'], tokenize=False)
            
            if not text:
                continue
                
            inputs = tokenizer(text, return_tensors='pt').to(device)
            if inputs["input_ids"].size(1) == 0:
                continue
                
            labels = inputs["input_ids"].clone()
            
            # Mask out the prompt
            seq = labels[0].tolist()
            found = False
            for j in range(len(seq) - response_len + 1):
                if seq[j:j+response_len] == response_token_ids:
                    labels[0, :j+response_len] = -100
                    found = True
                    break
                    
            if not found:
                labels[0, :] = -100
                
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Count only unmasked tokens for sequence length
            unmasked_len = (labels[0] != -100).sum().item()
            if unmasked_len > 0 and not torch.isnan(loss):
                total_nll += loss.item() * unmasked_len
                total_length += unmasked_len
            
            if i >= 100: # Evaluate on a subset to save time
                break

    if total_length == 0: return float('inf')
    return math.exp(total_nll / total_length)

def main():
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "data/models/qlora_adapter"
    dataset_path = "data/datasets/val.jsonl"
    
    print("Loading Dataset...")
    val_dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    print(f"Loading Base Model ({base_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # We load in 4bit to match training conditions and save VRAM
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    
    print("Calculating Perplexity for Base Model (No-LoRA)...")
    ppl_base = calculate_perplexity(base_model, tokenizer, val_dataset)
    print(f"Base Model Perplexity: {ppl_base:.4f}")
    
    print(f"Loading QLoRA Adapter ({adapter_path})...")
    qlora_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Calculating Perplexity for Fine-Tuned Model (QLoRA)...")
    ppl_qlora = calculate_perplexity(qlora_model, tokenizer, val_dataset)
    print(f"QLoRA Model Perplexity: {ppl_qlora:.4f}")
    
    # Save the results
    os.makedirs("plots/comparison", exist_ok=True)
    with open("plots/comparison/perplexity_results.txt", "w") as f:
        f.write(f"Base Model Perplexity: {ppl_base:.4f}\n")
        f.write(f"QLoRA Model Perplexity: {ppl_qlora:.4f}\n")

if __name__ == "__main__":
    main()
