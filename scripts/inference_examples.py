import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

def main():
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "data/models/qlora_adapter"
    dataset_path = "data/datasets/val.jsonl"
    
    print("Loading Base Model for Inference...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    
    print(f"Loading QLoRA Adapter ({adapter_path})...")
    qlora_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    val_dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    # We want 5-10 critical states. Let's find states mentioning "BUILD_BRIDGE" or "FIGHT_ENEMY"
    # or reflections.
    examples = []
    for item in val_dataset:
        if 'messages' in item:
            user_msg = item['messages'][1]['content']
            if 'BUILD_BRIDGE' in user_msg or 'FIGHT_ENEMY' in user_msg or 'REFLECTION REQUIRED' in user_msg:
                examples.append(item)
        if len(examples) >= 5:
            break
            
    print(f"Selected {len(examples)} examples for inference.")
    
    os.makedirs("plots/comparison", exist_ok=True)
    with open("plots/comparison/inference_qualitative_matrix.md", "w") as f:
        f.write("# Qualitative Matrix: NoLoRA vs LoRA\n\n")
        f.write("| State / Prompt Description | NoLoRA Action & Reasoning | LoRA Action & Reasoning |\n")
        f.write("|---|---|---|\n")
        
        for i, example in enumerate(examples):
            print(f"Processing example {i+1}/{len(examples)}...")
            prompt = tokenizer.apply_chat_template(example['messages'][:-1], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
            
            # Base Model Generation (NoLoRA)
            with qlora_model.disable_adapter():
                with torch.no_grad():
                    out_base = qlora_model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
            gen_base = tokenizer.decode(out_base[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # LoRA Model Generation
            with torch.no_grad():
                out_lora = qlora_model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
            gen_lora = tokenizer.decode(out_lora[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Extract state context for display
            user_msg = example['messages'][1]['content']
            state_desc = user_msg.replace('\n', '<br>').replace('|', '\\|')
            
            # Format generations for markdown table
            gen_base_fmt = gen_base.replace('\n', '<br>').replace('|', '\\|')
            gen_lora_fmt = gen_lora.replace('\n', '<br>').replace('|', '\\|')
            
            f.write(f"| <pre>{state_desc[:300]}...</pre> | <pre>{gen_base_fmt}</pre> | <pre>{gen_lora_fmt}</pre> |\n")
            
    print("Saved qualitative matrix to plots/comparison/inference_qualitative_matrix.md")

if __name__ == "__main__":
    main()
