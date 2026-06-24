"""
format_dataset.py — Convert Oracle Examples to ChatML JSONL
==============================================================

Converts the oracle_dataset.jsonl into the ChatML format required by Qwen 2.5:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import os
import random

def convert_to_chatml(example: dict) -> dict:
    prompt = example["prompt"]
    completion = example["completion"]
    
    # Simple split: everything before "### CURRENT STATE:" goes to system
    if "### CURRENT STATE:" in prompt:
        parts = prompt.split("### CURRENT STATE:")
        system_content = parts[0].strip()
        user_content = "### CURRENT STATE:\n" + parts[1].strip()
    else:
        system_content = "You are a helpful assistant orchestrating RL agents."
        user_content = prompt.strip()
        
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion}
        ]
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Format dataset for QLoRA fine-tuning")
    parser.add_argument("--input", type=str, default="data/datasets/raw/oracle_dataset.jsonl")
    parser.add_argument("--output", type=str, default="data/datasets")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    examples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(convert_to_chatml(json.loads(line)))

    # Since the new dataset generator already permutations over all agent statuses,
    # the dataset is naturally 550 examples. We do not need artificial inflation anymore.
    # Duplicate dataset to ensure enough gradient updates (augmentation)
    examples = examples * 10
    random.shuffle(examples)
    
    n_val = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.jsonl")
    val_path = os.path.join(args.output, "val.jsonl")

    for path, data in [(train_path, train_examples), (val_path, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")

    print(f"Train: {len(train_examples)} examples -> {train_path}")
    print(f"Val:   {len(val_examples)} examples -> {val_path}")

if __name__ == "__main__":
    main()
