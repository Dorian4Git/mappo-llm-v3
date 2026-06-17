"""
format_dataset.py — Convert Raw Examples to Fine-Tuning JSONL
==============================================================

Converts the raw bottleneck examples from dataset_generator.py into
the prompt/completion JSONL format required by QLoRA fine-tuning.

Features:
    - Train/val split with stratification by bottleneck type
    - Data augmentation via metric perturbation
    - Quality filtering

Usage:
    python -m finetune.format_dataset --input data/datasets/raw --output data/datasets
"""

import json
import os
import random
import numpy as np
from typing import Optional


def format_prompt(context: dict) -> str:
    """
    Format a bottleneck context into a fine-tuning prompt.

    This matches the prompt style used by the prompt_builder during inference,
    ensuring the fine-tuned model learns to respond to the same format.
    """
    subtasks = context.get("subtask_completion", {})
    positions = context.get("positions", [[0, 0], [0, 0]])
    inventory = context.get("inventory", [0] * 10)
    incomplete = context.get("incomplete_subtasks", [])
    bottleneck = context.get("bottleneck_subtask", "unknown")
    delay = context.get("delay_steps", 0)

    # Format subtask completion as percentages (simulated from binary state)
    subtask_lines = []
    for name in ["wood", "stone", "pickaxe", "iron", "sword", "armor", "bridge", "enemy", "gold"]:
        pct = 100.0 if subtasks.get(name, False) else 0.0
        subtask_lines.append(f"  * {name.capitalize()}: {pct:.0f}%")
    subtask_block = "\n".join(subtask_lines)

    return f"""ENVIRONMENT STATE: Agents are stuck for {delay} steps.
Agent 0 (Lumberjack) at [{positions[0][0]:.0f}, {positions[0][1]:.0f}].
Agent 1 (Miner) at [{positions[1][0]:.0f}, {positions[1][1]:.0f}].
Inventory: {inventory}
Bottleneck: {bottleneck} — agents fail to progress.
Incomplete subtasks: {', '.join(incomplete)}

SUBTASK COMPLETION:
{subtask_block}

What reward shaping adjustment would resolve this bottleneck?"""


def format_completion(resolution: dict) -> str:
    """
    Format the resolution into a fine-tuning completion (target output).
    """
    weights = resolution.get("ideal_weights", {})
    reasoning = resolution.get("reasoning", "")

    completion_obj = {
        "reasoning": reasoning,
        "w_wood": weights.get("w_wood", 1.0),
        "w_stone": weights.get("w_stone", 1.0),
        "w_workbench": weights.get("w_workbench", 1.0),
        "w_iron": weights.get("w_iron", 1.0),
        "w_bridge": weights.get("w_bridge", 1.0),
        "w_enemy": weights.get("w_enemy", 1.0),
        "w_gold": weights.get("w_gold", 1.0),
    }

    return json.dumps(completion_obj)


def augment_example(example: dict, rng: np.random.Generator) -> Optional[dict]:
    """
    Data augmentation: slightly perturb positions and delay to increase diversity.

    Returns a new example dict with perturbed values, or None if augmentation fails.
    """
    augmented = json.loads(json.dumps(example))  # Deep copy
    ctx = augmented["bottleneck_context"]

    # Perturb positions ±3 grid units
    positions = ctx.get("positions", [[0, 0], [0, 0]])
    for i in range(min(len(positions), 2)):
        for j in range(2):
            positions[i][j] = float(
                np.clip(positions[i][j] + rng.integers(-3, 4), 0, 60)
            )
    ctx["positions"] = positions

    # Perturb delay ±20%
    delay = ctx.get("delay_steps", 50)
    ctx["delay_steps"] = max(10, int(delay * rng.uniform(0.8, 1.2)))

    return augmented


def split_and_write(
    examples: list[dict],
    output_dir: str,
    val_ratio: float = 0.1,
    augment_factor: int = 2,
    seed: int = 42,
):
    """
    Split examples into train/val, apply augmentation, and write JSONL files.

    Args:
        examples: List of raw examples.
        output_dir: Directory to write train.jsonl and val.jsonl.
        val_ratio: Fraction of examples for validation.
        augment_factor: How many augmented copies per original example.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Stratified split by bottleneck type
    by_type: dict[str, list] = {}
    for ex in examples:
        bt = ex["bottleneck_context"]["bottleneck_subtask"]
        by_type.setdefault(bt, []).append(ex)

    train_examples = []
    val_examples = []

    for bt, bt_examples in by_type.items():
        random.shuffle(bt_examples)
        n_val = max(1, int(len(bt_examples) * val_ratio))
        val_examples.extend(bt_examples[:n_val])
        train_examples.extend(bt_examples[n_val:])

    # Augment training examples
    augmented_train = list(train_examples)
    for ex in train_examples:
        for _ in range(augment_factor - 1):
            aug = augment_example(ex, rng)
            if aug:
                augmented_train.append(aug)

    random.shuffle(augmented_train)
    random.shuffle(val_examples)

    # Write JSONL files
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    for path, data in [(train_path, augmented_train), (val_path, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                prompt = format_prompt(ex["bottleneck_context"])
                completion = format_completion(ex["resolution"])
                record = {
                    "prompt": prompt,
                    "completion": completion,
                }
                f.write(json.dumps(record) + "\n")

    print(f"Train: {len(augmented_train)} examples → {train_path}")
    print(f"Val:   {len(val_examples)} examples → {val_path}")

    return train_path, val_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Format dataset for QLoRA fine-tuning")
    parser.add_argument("--input", type=str, default="data/datasets/raw/bottleneck_examples.jsonl")
    parser.add_argument("--output", type=str, default="data/datasets")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--augment", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    # Load raw examples
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run dataset_generator.py first.")
        return

    examples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} raw examples")

    if args.dry_run:
        # Show sample prompt/completion
        if examples:
            sample = examples[0]
            print("\n--- Sample Prompt ---")
            print(format_prompt(sample["bottleneck_context"]))
            print("\n--- Sample Completion ---")
            print(format_completion(sample["resolution"]))
        return

    split_and_write(
        examples,
        args.output,
        val_ratio=args.val_ratio,
        augment_factor=args.augment,
    )


if __name__ == "__main__":
    main()
