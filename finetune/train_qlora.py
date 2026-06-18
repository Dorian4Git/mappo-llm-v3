"""
train_qlora.py — QLoRA Fine-Tuning for Bottleneck Expert LLM
==============================================================

Fine-tunes Qwen2.5-7B with QLoRA on the bottleneck→resolution dataset.
Produces a parameter-efficient adapter specialized in the crafting env mechanics.

Requirements:
    pip install transformers peft bitsandbytes datasets accelerate trl

Usage:
    python -m finetune.train_qlora --dataset data/datasets/train.jsonl
    python -m finetune.train_qlora --max-steps 1 --dataset data/datasets/train.jsonl  # dry run
"""

import os
import sys
import json
import argparse
import yaml


def load_llm_config() -> dict:
    """Load fine-tuning config from llm_config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "llm_config.yaml"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get("finetune", {})
    return {}


def train_qlora(
    dataset_path: str = "data/datasets/train.jsonl",
    val_path: str = "data/datasets/val.jsonl",
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "data/models/qlora_adapter",
    max_steps: int = -1,
    **kwargs,
):
    """
    Run QLoRA fine-tuning.

    Args:
        dataset_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.
        base_model: HuggingFace model ID.
        output_dir: Where to save the LoRA adapter.
        max_steps: Max training steps (-1 for full training).
    """
    # Late imports to avoid loading heavy libs unless actually fine-tuning
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    config = load_llm_config()
    base_model = kwargs.get("base_model", config.get("base_model", base_model))

    print(f"[QLoRA] Base model: {base_model}")
    print(f"[QLoRA] Dataset: {dataset_path}")
    print(f"[QLoRA] Output: {output_dir}")

    # ── Load Dataset ─────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={
        "train": dataset_path,
        "validation": val_path if os.path.exists(val_path) else dataset_path,
    })

    print(f"[QLoRA] Train examples: {len(dataset['train'])}")
    print(f"[QLoRA] Val examples: {len(dataset['validation'])}")

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantization Config (4-bit QLoRA) ────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load Model ───────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA Config ──────────────────────────────────────────────────
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.05)
    target_modules = config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training Arguments ───────────────────────────────────────────
    lr = config.get("learning_rate", 2e-4)
    epochs = config.get("epochs", 3)
    batch_size = 1  # Forced for 12GB VRAM
    grad_accum = 8  # Forced for 12GB VRAM
    warmup_ratio = config.get("warmup_ratio", 0.1)
    max_seq_len = 1024  # Forced for 12GB VRAM

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_steps=max_steps if max_steps > 0 else -1,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        gradient_checkpointing=True, # Forced for 12GB VRAM
        optim="paged_adamw_8bit",
        max_length=max_seq_len,
    )

    # ── Format Function ──────────────────────────────────────────────
    def formatting_func(examples):
        """Use Qwen's native ChatML template."""
        if isinstance(examples.get("messages", [None])[0], list):
            # Batched
            return [tokenizer.apply_chat_template(m, tokenize=False) for m in examples["messages"]]
        else:
            # Single example
            return tokenizer.apply_chat_template(examples["messages"], tokenize=False)

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # ── Train ────────────────────────────────────────────────────────
    print("[QLoRA] Starting training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[QLoRA] Adapter saved to {output_dir}")

    # Save training stats
    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[QLoRA] Eval metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for bottleneck expert LLM")
    parser.add_argument("--dataset", type=str, default="data/datasets/train.jsonl")
    parser.add_argument("--val", type=str, default="data/datasets/val.jsonl")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="data/models/qlora_adapter")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 for full). Use 1 for dry run.")
    args = parser.parse_args()

    train_qlora(
        dataset_path=args.dataset,
        val_path=args.val,
        base_model=args.base_model,
        output_dir=args.output,
        max_steps=args.max_steps,
    )
